import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_gather
from einops import rearrange
from typing import Tuple, Dict, List



ACT_LAYERS = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.1),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    'glu': nn.GLU(),
    None: nn.Identity(),
}

class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """
    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb



class PTA(nn.Module):
    def __init__(self, dim, num_heads, topks=[8,8]):
        super().__init__()
        self.dim = dim
        self.topks = topks
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)
        self.proj_i = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp_fine = nn.Sequential(nn.Linear(dim,dim), nn.GELU(), nn.Linear(dim,dim))
        self.fc = nn.Sequential(nn.Linear(dim,dim), nn.GELU(), nn.Linear(dim,dim))

    def full_attention(self, query, support):
        q = rearrange(self.proj_q(query), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(support), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(support), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) * self.scale
        attention_scores = torch.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        hidden_states = self.proj_o(hidden_states) + self.proj_i(query)
        attention_scores = attention_scores.mean(dim=1)

        return hidden_states, attention_scores
    
    def coarse_to_fine_ROI_selection(self, topk, support_len, query_inverse, support_index, attention, attention_indices=None):
        indices = torch.topk(attention, k=min(topk, attention.shape[-1])).indices  # (N', K)
        if attention_indices is not None:
            indices = torch.gather(attention_indices, dim=-1, index=indices)  # (N', K)
        support_index = torch.cat([support_index, torch.full_like(support_index[:1], support_len)])  # (M' + 1, L)
        indices = knn_gather(support_index.unsqueeze(0), indices.unsqueeze(0))  # (M' + 1, L), (N', K) -> (B, N', K, L)
        indices = rearrange(indices.squeeze(0), 'n k l -> n (k l)')  # (N', K * L)

        mask = torch.full((indices.shape[0], support_len+1), support_len, dtype=torch.long, device=indices.device)
        mask.scatter_(1, indices, indices)  # (N', M + 1)
        max_ROI_size = mask[:, :-1].lt(support_len).count_nonzero(dim=-1).max()
        indices = mask[:, :-1].topk(max_ROI_size.item(), largest=False, sorted=False).values  # (N', ?<=K*L)
        mask = indices.lt(support_len)[query_inverse]  # (N, ?<=K*L)
        return indices[query_inverse], mask

    def sparse_attention(self, query, support, attention_indices, attention_mask):
        q = rearrange(self.proj_q(query), 'b n (h c) -> b h n c', h=self.num_heads)
        k = torch.cat([self.proj_k(support), torch.zeros_like(support[:, :1])], dim=1)
        v = torch.cat([self.proj_v(support), torch.zeros_like(support[:, :1])], dim=1)
        k = rearrange(knn_gather(k, attention_indices), 'b n k (h c) -> b h n k c', h=self.num_heads)
        v = rearrange(knn_gather(v, attention_indices), 'b n k (h c) -> b h n k c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhnkc->bhnk', q, k) * self.scale
        attention_scores.masked_fill_(~attention_mask.unsqueeze(1), float('-inf'))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        hidden_states = torch.einsum('bhnk,bhnkc->bhnc', attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        hidden_states = self.proj_o(hidden_states) + self.proj_i(query)
        attention_scores = attention_scores.sum(dim=1)

        return hidden_states, attention_scores


class PointTreeSelfAttention(PTA):
    def __init__(self, dim, num_heads, topks=[8, 8]):
        super().__init__(dim, num_heads, topks)
    
    def forward(self, src_feats:List[torch.Tensor], src_tree:Dict[str,List[torch.Tensor]]) -> List[torch.Tensor]:
        src_feats_ = self.norm1(src_feats[-1].unsqueeze(0))
        src_feats_, src_attention = self.full_attention(src_feats_, src_feats_)
        src_feats_ = self.norm2(src_feats_).squeeze(0)
        src_attention_indices = None

        messages = [src_feats_]
        for l, topk in enumerate(self.topks):
            coarse_context = self.fc(messages[-1])
            coarse_context = torch.index_select(coarse_context, dim=0, index=src_tree['inverse'][-l-1])
            src_feats_ = src_feats[-l-2] + coarse_context
            src_attention_indices, src_attention_mask = self.coarse_to_fine_ROI_selection(
                topk,
                src_feats_.shape[0],
                src_tree['inverse'][-l-1],
                src_tree['index'][-l-1],
                src_attention.squeeze(0),
                src_attention_indices,
            )
            src_feats_, src_attention = self.sparse_attention(
                src_feats_.unsqueeze(0),
                src_feats_.unsqueeze(0),
                src_attention_indices.unsqueeze(0),
                src_attention_mask.unsqueeze(0)
            )
            src_feats_ = self.norm2(src_feats_).squeeze(0)
            src_feats_ = src_feats_ + self.mlp_fine(src_feats_) + coarse_context
            messages.append(src_feats_)
        
        messages.reverse()
        return messages



class PointTreeCrossAttention(PTA):
    def __init__(self, dim, num_heads, topks=[8, 8]):
        super().__init__(dim, num_heads, topks)
    
    def forward(self,
                src_feats:List[torch.Tensor],
                tgt_feats:List[torch.Tensor],
                src_tree:Dict[str,List[torch.Tensor]],
                tgt_tree:Dict[str,List[torch.Tensor]]
                ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        src_feats_ = self.norm1(src_feats[-1].unsqueeze(0))
        tgt_feats_ = self.norm1(tgt_feats[-1].unsqueeze(0))

        src_feats_new, src_attention = self.full_attention(src_feats_, tgt_feats_)
        tgt_feats_new, tgt_attention = self.full_attention(tgt_feats_, src_feats_)
        
        src_feats_ = self.norm2(src_feats_new).squeeze(0)
        tgt_feats_ = self.norm2(tgt_feats_new).squeeze(0)

        src_attention_indices = None
        tgt_attention_indices = None

        src_messages = [src_feats_]
        tgt_messages = [tgt_feats_]

        for l, topk in enumerate(self.topks):
            src_coarse_context, tgt_coarse_context = self.fc(src_messages[-1]), self.fc(tgt_messages[-1])
            src_coarse_context = torch.index_select(src_coarse_context, dim=0, index=src_tree['inverse'][-l-1])
            tgt_coarse_context = torch.index_select(tgt_coarse_context, dim=0, index=tgt_tree['inverse'][-l-1])
            src_feats_ = src_feats[-l-2] + src_coarse_context
            tgt_feats_ = tgt_feats[-l-2] + tgt_coarse_context

            src_attention_indices, src_attention_mask = self.coarse_to_fine_ROI_selection(
                topk,
                tgt_feats_.shape[0],
                src_tree['inverse'][-l-1],
                tgt_tree['index'][-l-1],
                src_attention.squeeze(0),
                src_attention_indices,
            )
            tgt_attention_indices, tgt_attention_mask = self.coarse_to_fine_ROI_selection(
                topk,
                src_feats_.shape[0],
                tgt_tree['inverse'][-l-1],
                src_tree['index'][-l-1],
                tgt_attention.squeeze(0),
                tgt_attention_indices,
            )
            src_feats_new, src_attention = self.sparse_attention(
                src_feats_.unsqueeze(0),
                tgt_feats_.unsqueeze(0),
                src_attention_indices.unsqueeze(0),
                src_attention_mask.unsqueeze(0)
            )
            tgt_feats_new, tgt_attention = self.sparse_attention(
                tgt_feats_.unsqueeze(0),
                src_feats_.unsqueeze(0),
                tgt_attention_indices.unsqueeze(0),
                tgt_attention_mask.unsqueeze(0)
            )
            src_feats_ = self.norm2(src_feats_new).squeeze(0)
            src_feats_ = src_feats_ + self.mlp_fine(src_feats_) + src_coarse_context
            src_messages.append(src_feats_)

            tgt_feats_ = self.norm2(tgt_feats_new).squeeze(0)
            tgt_feats_ = tgt_feats_ + self.mlp_fine(tgt_feats_) + tgt_coarse_context
            tgt_messages.append(tgt_feats_)
        
        src_messages.reverse()
        tgt_messages.reverse()
        return src_messages, tgt_messages


class TreeTransformerCrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, topks, activation="relu", rpe=False):
        super().__init__()
        self.rpe = rpe
        self.sa = PointTreeSelfAttention(d_model, nhead, topks)
        self.ca = PointTreeCrossAttention(d_model, nhead, topks)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            ACT_LAYERS[activation],
            nn.Linear(dim_feedforward, d_model),
        )
        self.pool_mlp_before_self_attention = nn.Sequential(
            nn.Linear(d_model+3, dim_feedforward),
            ACT_LAYERS[activation],
            nn.Linear(dim_feedforward, d_model),
            nn.LayerNorm(d_model),
        )
        self.pool_mlp_before_cross_attention = nn.Sequential(
            nn.Linear(d_model+3, dim_feedforward),
            ACT_LAYERS[activation],
            nn.Linear(dim_feedforward, d_model),
            nn.LayerNorm(d_model),
        )

    def pool(self, net:nn.Sequential, feature:torch.Tensor, tree:Dict[str,List[torch.Tensor]], messages=None):
        """
        concat the relative coordinates and features of points in a voxel, feed forward a point-wise MLP, AvgPool
        maybe we can further fuse the pooled features with those from the former layerss
        """
        feature_list = [feature]
        for i, inverse in enumerate(tree['inverse']):
            pcd_fine = tree['points'][i]
            pcd_coarse = tree['points'][i + 1][inverse]
            feature = net(torch.cat([feature, pcd_fine - pcd_coarse], dim=-1))  # (N, C)
            feature = torch.cat([feature, torch.zeros_like(feature[:1])]).unsqueeze(0)  # (1, N+1, C)
            feature = knn_gather(feature, tree['index'][i].unsqueeze(0)).squeeze(0).sum(1)  # (K, C)
            feature = feature / tree['counts'][i].float().unsqueeze(-1)  # (K, 3)
            #feature = torch_scatter.scatter_mean(feature, inverse, dim=0)
            if messages is not None: feature = messages[i+1] + feature
            feature_list.append(feature)
        return feature_list
    
    def forward(self, src, tgt, src_tree:Dict[str,List[torch.Tensor]], tgt_tree:Dict[str,List[torch.Tensor]]):
        src = self.norm1(src)
        src2 = src + src_tree['ape'][0]
        src_feats = self.pool(self.pool_mlp_before_self_attention, src2, src_tree)
        src_messages = self.sa.forward(src_feats, src_tree)
        src = src + src_messages[0]
        
        tgt = self.norm1(tgt)
        tgt2 = tgt + tgt_tree['ape'][0]
        tgt_feats = self.pool(self.pool_mlp_before_self_attention, tgt2, tgt_tree)
        tgt_messages = self.sa.forward(tgt_feats, tgt_tree)
        tgt = tgt + tgt_messages[0]

        src = self.norm2(src)
        src3 = src + src_tree['ape'][0]
        tgt = self.norm2(tgt)
        tgt3 = tgt + tgt_tree['ape'][0]
        src_feats = self.pool(self.pool_mlp_before_cross_attention, src3, src_tree, src_messages)
        tgt_feats = self.pool(self.pool_mlp_before_cross_attention, tgt3, tgt_tree, tgt_messages)
        src_messages, tgt_messages = self.ca.forward(src_feats, tgt_feats, src_tree, tgt_tree)

        src = src + src_messages[0] #self.linear_ca(src3)  # 
        tgt = tgt + tgt_messages[0] #self.linear_ca(tgt3)  # 

        src = src + self.ffn(src)
        tgt = tgt + self.ffn(tgt)

        return src, tgt



class TreeTransformerCrossEncoder(nn.Module):
    def __init__(self, cfg, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TreeTransformerCrossEncoderLayer(cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim, cfg.topks)
            for _ in range(cfg.blocks)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.return_intermediate = return_intermediate

        self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.hidden_dim)
        #self.gs_embed = nn.ModuleList([GeometricStructureEmbedding(
        #    cfg.hidden_dim, cfg.sigma_d, cfg.sigma_a, cfg.angle_k, cfg.reduction_a
        #)] for _ in range(cfg.pyramid_levels))

    def forward(self, src, tgt, src_tree:Dict[str,List[torch.Tensor]], tgt_tree:Dict[str,List[torch.Tensor]]):
        src_tree['ape'] = [self.pos_embed(src_tree['points'][0])] #[self.pos_embed(v) for v in src_tree['points']]
        tgt_tree['ape'] = [self.pos_embed(tgt_tree['points'][0])] #[self.pos_embed(v) for v in tgt_tree['points']]
        #src_tree['gse'] = [self.gs_embed[i](v) for i,v in enumerate(src_tree['points'])]
        #tgt_tree['gse'] = [self.gs_embed[i](v) for i,v in enumerate(tgt_tree['points'])]

        src_intermediate, tgt_intermediate = [], []

        for layer in self.layers:
            src, tgt = layer(src, tgt, src_tree, tgt_tree)
            src_intermediate.append(self.norm(src) if self.norm is not None else src)
            tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src_intermediate[-1], tgt_intermediate[-1]



class CorrespondenceRegressor(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3)
        )
        self.conf_logits_decoder = nn.Linear(d_embed, 1)

    def forward(self, feats_padded):
        corr = self.coor_mlp(feats_padded)
        overlap = self.conf_logits_decoder(feats_padded)
        return corr, overlap
