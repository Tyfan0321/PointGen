import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch3d.ops import knn_points, knn_gather

from models.transformer.vanilla_transformer import MultiHeadAttention
from models.transformer.positional_encoding import GeometricStructureEmbedding
from models.cast.regtr import TransformerCrossEncoderLayer, PositionEmbeddingCoordsSine
from models.cast.spot_attention import Upsampling, Downsampling


class SpotGuidedTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, activation_fn='relu'):
        super(SpotGuidedTransformerLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    @torch.no_grad()
    def select_spots(self, input_knn, memory_knn, confidence_scores, matching_indices, num_spots):
        """
        Args:
            input_knn (Tensor): (B, N, k+1)
            memory_knn (Tensor): (B, M, K)
            confidence_scores (Tensor): (B, N, 1)
            matching_indices (Tensor): (B, N, 1)
            num_spots (int):= S

        Returns:
            spot_mask: torch.Tensor (B, N, <=(S+1)*K)
            spot_indices: torch.Tensor (B, N, <=(S+1)*K)
        """
        knn_scores = knn_gather(confidence_scores, input_knn[...,1:]).squeeze(-1)  # (B, N, k)
        confidence_scores, confident_knn = knn_scores.topk(k=num_spots)  # (B, N, S)
        confident_knn = torch.gather(input_knn[...,1:], -1, confident_knn)  # (B, N, S)
        confident_knn = torch.cat([input_knn[...,:1], confident_knn], dim=-1)  # (B, N, S+1)
        
        spot_indices = knn_gather(matching_indices, confident_knn).squeeze(-1)  # (B, N, S+1)
        spot_indices = knn_gather(memory_knn, spot_indices)  # (B, N, S+1, K)
        spot_indices = rearrange(spot_indices, 'b n s k -> b n (s k)')  # (B, N, (S+1)*K)
        
        # avoid redundant indices from spot areas
        B, N, M = input_knn.shape[0], input_knn.shape[1], memory_knn.shape[1]
        attention_mask = torch.zeros((B, N, M), device=input_knn.device, dtype=torch.int)
        attention_mask.scatter_(-1, spot_indices, 1)  # (B, N, M)
        spot_mask, spot_indices = attention_mask.topk(attention_mask.sum(dim=-1).max().item())  # (B, N, ?)
        return spot_mask.bool(), spot_indices
    
    def spot_guided_attention(self, input_states, memory_states, indices, attention_mask=None):
        q = self.proj_q(input_states)  # (B, N, H*C)
        k = knn_gather(self.proj_k(memory_states), indices)  # (B, N, K, H*C)
        v = knn_gather(self.proj_v(memory_states), indices)  # (B, N, K, H*C)
        
        q = rearrange(q, 'b n (h c) -> b h n c', h=self.num_heads)  # (B, H, N, C)
        k = rearrange(k, 'b n m (h c) -> b h n m c', h=self.num_heads)  # (B, H, N, K, C)
        v = rearrange(v, 'b n m (h c) -> b h n m c', h=self.num_heads)  # (B, H, N, K, C)
        
        attention_scores = torch.einsum('bhnc,bhnmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_mask is not None:
            attention_scores.masked_fill_(~attention_mask.unsqueeze(1), float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.sum(attention_scores.unsqueeze(-1) * v, dim=-2)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        return hidden_states
    
    def forward(self,
                input_states,
                memory_states,
                input_pe,
                memory_pe,
                input_spot_indices,
                input_spot_mask,
                memory_spot_indices,
                memory_spot_mask,
                input_self_attention_token_ids,
                memory_self_attention_token_ids,
                ):
        
        input_states_w_pe = self.norm1(input_states) + input_pe
        input_states_kv = knn_gather(input_states_w_pe, input_self_attention_token_ids.unsqueeze(1)).squeeze(1)
        input_states_w_pe,_ = self.self_attention(input_states_w_pe, input_states_kv, input_states_kv)
        input_states = input_states + input_states_w_pe

        memory_states_w_pe = self.norm1(memory_states) + memory_pe
        memory_states_kv = knn_gather(memory_states_w_pe, memory_self_attention_token_ids.unsqueeze(1)).squeeze(1)
        memory_states_w_pe,_ = self.self_attention(memory_states_w_pe, memory_states_kv, memory_states_kv)
        memory_states = memory_states + memory_states_w_pe

        input_states_w_pe = self.norm2(input_states) + input_pe
        memory_states_w_pe = self.norm2(memory_states) + memory_pe
        input_states_w_pe = self.spot_guided_attention(input_states_w_pe, memory_states_w_pe, input_spot_indices, input_spot_mask)
        memory_states_w_pe = self.spot_guided_attention(memory_states_w_pe, input_states_w_pe, memory_spot_indices, memory_spot_mask)

        # Position-wise feedforward
        input_states = input_states + input_states_w_pe
        memory_states = memory_states + memory_states_w_pe
        input_states = input_states + self.linear2(F.relu(self.linear1(self.norm3(input_states_w_pe))))
        memory_states = memory_states + self.linear2(F.relu(self.linear1(self.norm3(memory_states_w_pe))))

        return input_states, memory_states


class SpotGuidedGeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(SpotGuidedGeoTransformer, self).__init__()
        self.k = cfg.k            # num of neighbor points whose corresponding patches are candidate spots
        self.spots = cfg.spots    # num of neighbor points whose corresponding patches are selected as spots
        self.down_k = cfg.down_k  # num of nodes for down sampling and fusion with semi-dense points
        self.spot_k = cfg.spot_k  # num of points in a spot
        self.full_blocks = cfg.full_blocks
        self.cast_blocks = cfg.cast_blocks
        self.sigma_c2 = cfg.sigma_c * cfg.sigma_c
        self.seed_threshold = cfg.seed_threshold
        self.dual_normalization = cfg.dual_normalization

        self.in_proj1 = nn.Linear(cfg.input_dim_c, cfg.hidden_dim)
        self.in_proj2 = nn.Linear(cfg.input_dim_f, cfg.hidden_dim)

        self.regtr_layers = nn.ModuleList([
            TransformerCrossEncoderLayer(cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim, rpe=False) \
            for _ in range(self.full_blocks)
        ])

        self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.hidden_dim)
        if "sigma_d" in cfg.keys() and "sigma_a" in cfg.keys():
            self.embed = GeometricStructureEmbedding(
                cfg.hidden_dim, cfg.sigma_d, cfg.sigma_a, cfg.angle_k, cfg.reduction_a
            )
            self.geometric_structure_embedding = True
        else: self.geometric_structure_embedding = False
        
        self.coarse_layers = nn.ModuleList()
        self.fine_layers = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.downsampling = nn.ModuleList()

        for _ in range(self.cast_blocks):
            self.upsampling.append(Upsampling(cfg.hidden_dim, cfg.hidden_dim))
            self.downsampling.append(Downsampling(cfg.hidden_dim, cfg.hidden_dim))
            self.coarse_layers.append(TransformerCrossEncoderLayer(
                cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim, rpe=self.geometric_structure_embedding
            ))
            self.fine_layers.append(SpotGuidedTransformerLayer(
                cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim, cfg.activation_fn
            ))
        
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.W = torch.nn.Parameter(torch.zeros(self.cast_blocks, cfg.hidden_dim, cfg.hidden_dim))
        torch.nn.init.normal_(self.W, std=0.1)
    
    def matching_scores(self, ref_feats:torch.Tensor, src_feats:torch.Tensor, W:torch.Tensor):
        W_triu = torch.triu(W)
        W_symmetrical = W_triu + W_triu.T
        matching_scores = torch.einsum('...ic,cd,...jd->...ij', ref_feats, W_symmetrical, src_feats)
        if self.dual_normalization:
            ref_matching_scores = torch.softmax(matching_scores, dim=-1)
            src_matching_scores = torch.softmax(matching_scores, dim=-2)
            return matching_scores, ref_matching_scores * src_matching_scores
        return matching_scores
    
    @torch.no_grad()
    def compatibility_scores(self, ref_dists, src_dists, matching_indices):
        """
        Args:
            ref_dists (Tensor): (B, N, N)
            src_dists (Tensor): (B, M, M)
            matching_indices (Tensor): (B, N, 1)

        Returns:
            compatibility (Tensor): (B, N, N)
        """
        src_dists = knn_gather(src_dists, matching_indices).squeeze(2)  # (B, N, 1, M)
        src_dists = knn_gather(src_dists.transpose(1, 2), matching_indices).squeeze(2)  # (B, N, N)
        return torch.relu(1. - torch.square(ref_dists - src_dists) / self.sigma_c2)  # (B, N, N)
    
    @torch.no_grad()
    def seeding(self, compatible_scores:torch.Tensor, confidence_scores:torch.Tensor):
        selection_scores = compatible_scores.gt(compatible_scores.max(-1, True)[0] * self.seed_threshold)  # (B, N)
        if selection_scores.shape[0] == 1:
            return selection_scores[0].nonzero().view(1,-1)
        seed_num = selection_scores.count_nonzero(1).min().item()
        return confidence_scores.masked_fill(~selection_scores,0.).topk(seed_num, dim=-1).indices  # (B, K)
    
    def forward(self,ref_points,src_points, ref_feats,src_feats, ref_points_c,src_points_c, ref_feats_c,src_feats_c):
        """
        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_points_c (Tensor): (B, N', 3)
            src_points_c (Tensor): (B, M', 3)
            ref_feats_c (Tensor): (B, N', C')
            src_feats_c (Tensor): (B, M', C')

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
            matching_scores: List[torch.Tensor] (B, N, M)
        """
        ref_pos_emb_c = self.pos_embed(ref_points_c)
        src_pos_emb_c = self.pos_embed(src_points_c)

        ref_pos_emb = self.pos_embed(ref_points)
        src_pos_emb = self.pos_embed(src_points)

        ref_feats_c = self.in_proj1(ref_feats_c)
        src_feats_c = self.in_proj1(src_feats_c)
        
        ref_feats = self.in_proj2(ref_feats)
        src_feats = self.in_proj2(src_feats)

        for layer in self.regtr_layers:
            ref_feats, src_feats = layer(ref_feats, src_feats, ref_pos_emb, src_pos_emb)
        new_ref_feats, new_src_feats = ref_feats, src_feats
        
        k = max(self.k + 1, self.spot_k)
        with torch.no_grad():
            ref_dists = torch.cdist(ref_points, ref_points)  # (B, N, N)
            src_dists = torch.cdist(src_points, src_points)  # (B, M, M)
            ref_idx = ref_dists.topk(k, largest=False).indices  # (B, N, k)
            src_idx = src_dists.topk(k, largest=False).indices  # (B, M, k)

            # for nearest up-sampling fusion
            ref_idx_up = knn_points(ref_points, ref_points_c)[1]  # (B, N, 1)
            src_idx_up = knn_points(src_points, src_points_c)[1]  # (B, M, 1)
        
        # for knn interpolation in down-sampling fusion
        _, ref_idx_down, ref_xyz_down = knn_points(ref_points_c, ref_points, K=self.down_k, return_nn=True)
        _, src_idx_down, src_xyz_down = knn_points(src_points_c, src_points, K=self.down_k, return_nn=True)
        
        if self.geometric_structure_embedding:
            ref_embeddings = self.embed(ref_points_c)
            src_embeddings = self.embed(src_points_c)

        correlation = []

        for i in range(self.cast_blocks):
            if self.geometric_structure_embedding:
                new_ref_feats_c, new_src_feats_c = self.coarse_layers[i](
                    ref_feats_c, src_feats_c, ref_pos_emb_c, src_pos_emb_c, ref_embeddings, src_embeddings
                )
            else:
                new_ref_feats_c, new_src_feats_c = self.coarse_layers[i](
                    ref_feats_c, src_feats_c, ref_pos_emb_c, src_pos_emb_c
                )
            
            ref_feats = self.upsampling[i](new_ref_feats, new_ref_feats_c, ref_idx_up)
            src_feats = self.upsampling[i](new_src_feats, new_src_feats_c, src_idx_up)
            
            ref_feats_c = self.downsampling[i](new_ref_feats_c, new_ref_feats, ref_points_c, ref_xyz_down, ref_idx_down)
            src_feats_c = self.downsampling[i](new_src_feats_c, new_src_feats, src_points_c, src_xyz_down, src_idx_down)

            if self.dual_normalization:
                raw_matching_scores, matching_scores = self.matching_scores(ref_feats, src_feats, self.W[i])
                correlation.append(raw_matching_scores)
            else:
                matching_scores = self.matching_scores(ref_feats, src_feats, self.W[i])
                correlation.append(matching_scores)

            confidence_scores, matching_indices = torch.max(matching_scores, dim=-1, keepdim=True)
            compatible_scores = self.compatibility_scores(ref_dists, src_dists, matching_indices).mean(-1)
            confidence_scores = confidence_scores * compatible_scores.unsqueeze(-1)
            ref_token_indices = self.seeding(compatible_scores, confidence_scores)
            ref_spot_mask, ref_spot_indices = self.fine_layers[i].select_spots(
                ref_idx[..., :self.k+1], src_idx[..., :self.spot_k], confidence_scores, matching_indices, self.spots
            )
            
            confidence_scores, matching_indices = torch.max(matching_scores.transpose(1, 2), dim=-1, keepdim=True)
            compatible_scores = self.compatibility_scores(src_dists, ref_dists, matching_indices).mean(-1)
            confidence_scores = confidence_scores * compatible_scores.unsqueeze(-1)
            src_token_indices = self.seeding(compatible_scores, confidence_scores)
            src_spot_mask, src_spot_indices = self.fine_layers[i].select_spots(
                src_idx[..., :self.k+1], ref_idx[..., :self.spot_k], confidence_scores, matching_indices, self.spots
            )

            new_ref_feats,new_src_feats = self.fine_layers[i](
                ref_feats, src_feats,
                ref_pos_emb, src_pos_emb,
                ref_spot_indices, ref_spot_mask,
                src_spot_indices, src_spot_mask, 
                ref_token_indices, src_token_indices,
            )
        
        new_ref_feats, new_src_feats = self.norm(ref_feats), self.norm(src_feats)

        return new_ref_feats, new_src_feats, correlation
