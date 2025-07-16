import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_gather
from einops import rearrange

from models.transformer.output_layer import AttentionOutput
from models.transformer.positional_encoding import RotaryPositionalEmbedding
from models.kpconv import UnaryBlock


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampling, self).__init__()
        self.unary = nn.Sequential(
            UnaryBlock(in_channels, out_channels),
            UnaryBlock(out_channels, out_channels)
        )
        self.output = UnaryBlock(out_channels, out_channels)
    
    def forward(self, query, support, upsample_indices):
        """
        Args:
            query (Tensor): (B, N, C)
            support (Tensor): (B, M, C')
            upsample_indices (Tensor): (B, N, 1)
        return:
            latent (Tensor): (B, N, C)
        """
        latent = knn_gather(support, upsample_indices).squeeze(2)
        return self.output(self.unary(latent) + query)


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampling, self).__init__()
        self.unary = nn.Sequential(
            UnaryBlock(in_channels, out_channels),
            UnaryBlock(out_channels, out_channels)
        )
        self.output = UnaryBlock(out_channels, out_channels)
    
    def forward(self, q_feats, s_feats, q_points:torch.Tensor, s_points:torch.Tensor, downsample_indices):
        """
        Args:
            q_feats (Tensor): (B, N, C)
            s_feats (Tensor): (B, M, C')
            q_points (Tensor): (B, N, 3)
            s_points (Tensor): (B, N, K, 3)
            downsample_indices (Tensor): (B, N, K)
        return:
            latent (Tensor): (B, M, C)
        """
        grouped_feats = knn_gather(s_feats, downsample_indices) # (B, N, K, C')
        knn_weights = 1. / ((s_points - q_points.unsqueeze(2)).pow(2).sum(-1) + 1e-8) # (B, N, K)
        knn_weights = knn_weights / knn_weights.sum(dim=-1, keepdim=True) # (B, N, K)
        latent = torch.sum(grouped_feats * knn_weights.unsqueeze(-1), dim=2) # (B, N, C)
        return self.output(self.unary(latent) + q_feats)


class SparseTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, pe=True, dropout=None, activation_fn='relu'):
        super(SparseTransformerLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.pe = pe
        
        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output = AttentionOutput(d_model, dropout, activation_fn)
        if pe: self.rpe = RotaryPositionalEmbedding(self.d_model)
    
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
            spot_mask: torch.Tensor (B, N, (S+1)*K)
            spot_indices: torch.Tensor (B, N, (S+1)*K)
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
        spot_mask, spot_indices = attention_mask.topk(min(spot_indices.shape[-1],attention_mask.shape[-1]))  # (B, N, (S+1)*K)
        return spot_mask.bool(), spot_indices
    
    def forward(self, input_states, memory_states, indices=None, input_coord=None, memory_coord=None, attention_mask=None):
        """Sparse Transformer Layer

        Args:
            input_states (Tensor): (B, N, C)
            memory_states (Tensor): (B, M, C)
            indices (Tensor): (B, N, K) | (B, K)
            input_coord (Tensor): (B, N, 3)
            memory_coord (Tensor): (B, M, 3)
            attention_mask (Tensor): (B, N, K)

        Returns:
            output_states: torch.Tensor (B, N, C)
        """
        q = self.proj_q(input_states)  # (B, N, H*C)
        
        if indices is not None and indices.ndim == 3:
            k = knn_gather(self.proj_k(memory_states), indices)  # (B, N, K, H*C)
            v = knn_gather(self.proj_v(memory_states), indices)  # (B, N, K, H*C)
            
            if self.pe and memory_coord is not None and input_coord is not None:
                k = self.rpe(knn_gather(memory_coord, indices) - input_coord.unsqueeze(2), k)
            
            q = rearrange(q, 'b n (h c) -> b h n c', h=self.num_heads)  # (B, H, N, C)
            k = rearrange(k, 'b n m (h c) -> b h n m c', h=self.num_heads)  # (B, H, N, K, C)
            v = rearrange(v, 'b n m (h c) -> b h n m c', h=self.num_heads)  # (B, H, N, K, C)
            attention_scores = torch.einsum('bhnc,bhnmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        
        else:
            if indices is not None:
                assert indices.ndim == 2
                memory_states = knn_gather(memory_states, indices.unsqueeze(1)).squeeze(1)  # (B, K, C)
                if self.pe and memory_coord is not None and input_coord is not None:
                    memory_coord = knn_gather(memory_coord, indices.unsqueeze(1)).squeeze(1)  # (B, K, 3)
            
            k, v = self.proj_k(memory_states), self.proj_v(memory_states)  # (B, K, H*C)
            if self.pe and memory_coord is not None and input_coord is not None:
                q, k = self.rpe(input_coord, q), self.rpe(memory_coord, k)  # (B, K, H*C)
            
            q = rearrange(q, 'b n (h c) -> b h n c', h=self.num_heads)  # (B, H, N, C)
            k = rearrange(k, 'b k (h c) -> b h k c', h=self.num_heads)  # (B, H, K, C)
            v = rearrange(v, 'b k (h c) -> b h k c', h=self.num_heads)  # (B, H, K, C)
            attention_scores:torch.Tensor = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        
        if attention_mask is not None:
            attention_scores.masked_fill_(~attention_mask.unsqueeze(1), float('-inf'))
            #attention_scores = attention_scores - 1e6 * (1. - attention_mask.unsqueeze(1))
        attention_scores = F.softmax(attention_scores, dim=-1)
        if indices is None or indices.ndim == 2:
            hidden_states = torch.einsum('bhnm,bhmc->bhnc', attention_scores, v)
        else:
            hidden_states = torch.sum(attention_scores.unsqueeze(-1) * v, dim=-2)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        output_states = self.output(output_states)
        return output_states



class ExpertNet(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))



class GuidedDeepSeekMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts=16, num_shared_experts=2, guidance_bins=20, 
                 top_k=2, balance_update_rate=0.01, min_bias=-2.0, max_bias=2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.guidance_resolution = 1./guidance_bins

        self.balance_update_rate = balance_update_rate
        self.min_bias = min_bias
        self.max_bias = max_bias
        
        # 专家网络
        '''self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_experts)
        ])'''
        self.experts = nn.ModuleList([ExpertNet(hidden_dim, hidden_dim * 2) for _ in range(self.num_experts)])
        self.shared_experts = ExpertNet(hidden_dim, self.num_shared_experts * hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 门控网络
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.guidance_emb_1 = nn.Embedding(guidance_bins*5, self.hidden_dim//2)
        self.guidance_emb_2 = nn.Embedding(guidance_bins, self.hidden_dim//2)

        nn.init.ones_(self.guidance_emb_1.weight)
        nn.init.ones_(self.guidance_emb_2.weight)
        
        # 自适应专家偏置 (不通过梯度更新)
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        
        # 负载统计 (EMA)
        self.register_buffer("ema_utilization", torch.zeros(num_experts))
        self.register_buffer("ema_total_tokens", torch.tensor(0.0))
        self.ema_decay = 0.99

    def forward(self, input_states:torch.Tensor, input_guidance:torch.Tensor=None):
        batch_size, seq_len, _ = input_states.shape  # (B, N, C)
        x_flat = input_states.view(-1, self.hidden_dim)  # (B*N, C)
        num_tokens = x_flat.size(0)
        
        # 1. 计算门控分数 (加入专家偏置)
        if input_guidance is not None:
            emb_1 = self.guidance_emb_1((input_guidance[:,:,0] / (self.guidance_resolution/5.)).long())
            emb_2 = self.guidance_emb_2((input_guidance[:,:,1] / self.guidance_resolution).long())
            guidance_emb = torch.cat([emb_1, emb_2], dim=-1).view(-1, self.hidden_dim)  # (B*N, C)
            original_gate_logits:torch.Tensor = self.gate(x_flat * guidance_emb)  # (B*N, E)
        else:
            original_gate_logits:torch.Tensor = self.gate(x_flat)  # (B*N, E)
        gate_logits = original_gate_logits + self.expert_bias  # (B*N, E)
        
        # 2. 选择top-k专家
        top_k_gate, top_k_indices = torch.topk(original_gate_logits, k=self.top_k, dim=-1)  # (B*N, k)
        
        # 3. 计算专家容量
        #expert_capacity = int(self.capacity_factor * num_tokens / self.num_experts)
        #expert_capacity = max(expert_capacity, 1)
        
        # 4. 创建路由掩码
        mask = torch.zeros_like(gate_logits, dtype=torch.bool)
        expert_counts = torch.zeros(self.num_experts, device=input_states.device)
        
        # 5. 动态路由分配
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (B*N,)
            candidate_tokens = torch.nonzero(expert_mask).squeeze(-1)  # (B*N,)
            num_candidates = len(candidate_tokens)
            expert_counts[expert_idx] = num_candidates
            mask[candidate_tokens, expert_idx] = True
        
        # 6. 重新计算有效门控分数
        original_gate_logits.masked_fill_(~mask, float("-inf"))
        top_k_gate, top_k_indices = torch.topk(original_gate_logits, k=self.top_k, dim=-1)  # (B*N, k)
        gate_scores = F.softmax(top_k_gate, dim=-1)  # (B*N, k)
        
        # 7. 专家计算
        hidden_states = torch.zeros_like(x_flat)  # (B*N, C)
        expert_outputs = [None] * self.num_experts
        
        self._update_load_statistics(expert_counts, num_tokens)
        self._update_expert_biases()
        
        for expert_idx in range(self.num_experts):
            token_indices = torch.nonzero(mask[:, expert_idx]).squeeze(-1)
            if len(token_indices) > 0:
                expert_output = self.experts[expert_idx](x_flat.index_select(0,token_indices))
                expert_outputs[expert_idx] = (token_indices, expert_output)
        
        # 8. 聚合输出
        for k in range(self.top_k):
            expert_idx_for_token = top_k_indices[:, k]
            for expert_idx in expert_idx_for_token.unique():
                exp_tokens = expert_idx_for_token.eq(expert_idx).nonzero().squeeze(-1)
                if expert_outputs[expert_idx] is not None:
                    src_indices, expert_out = expert_outputs[expert_idx]
                    src_indices = torch.nonzero(src_indices[:, None] == exp_tokens)[:, 0]
                    weights = gate_scores.index_select(0, exp_tokens)[:, k].unsqueeze(-1)
                    hidden_states.index_add_(0, exp_tokens, weights * expert_out.index_select(0, src_indices))
        
        return self.norm(input_states + hidden_states.view(batch_size, seq_len, self.hidden_dim))
    
    def _update_load_statistics(self, expert_counts, num_tokens):
        """更新专家负载统计 (指数移动平均)"""
        if self.training:
            utilization = expert_counts / (expert_counts.sum() + 1e-7)
            self.ema_utilization = self.ema_decay * self.ema_utilization + (1. - self.ema_decay) * utilization
            self.ema_total_tokens = self.ema_decay * self.ema_total_tokens + (1. - self.ema_decay) * num_tokens
    
    @torch.no_grad()
    def _update_expert_biases(self):
        """基于负载均衡动态更新专家偏置"""
        if self.training and self.ema_total_tokens > 100:
            imbalance = self.ema_utilization * self.num_experts - 1.0
            bias_update = self.balance_update_rate * torch.log1p(torch.abs(imbalance)) * torch.sign(imbalance)
            new_bias = self.expert_bias + bias_update
            new_bias = torch.clamp(new_bias, self.min_bias, self.max_bias)
            self.expert_bias.copy_(new_bias)



class GuidedTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, guidance_bins:int, pe=True, dropout=None, activation_fn='relu'):
        super(GuidedTransformerLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.guidance_resolution = 1. / guidance_bins
        self.pe = pe
        
        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.guidance_emb_q1 = nn.Embedding(guidance_bins*5, self.d_model//2)
        self.guidance_emb_k1 = nn.Embedding(guidance_bins*5, self.d_model//2)
        self.guidance_emb_v1 = nn.Embedding(guidance_bins*5, self.d_model//2)

        self.guidance_emb_q2 = nn.Embedding(guidance_bins, self.d_model//2)
        self.guidance_emb_k2 = nn.Embedding(guidance_bins, self.d_model//2)
        self.guidance_emb_v2 = nn.Embedding(guidance_bins, self.d_model//2)

        nn.init.zeros_(self.guidance_emb_q1.weight)
        nn.init.zeros_(self.guidance_emb_k1.weight)
        nn.init.ones_(self.guidance_emb_v1.weight)

        nn.init.zeros_(self.guidance_emb_q2.weight)
        nn.init.zeros_(self.guidance_emb_k2.weight)
        nn.init.ones_(self.guidance_emb_v2.weight)
        
        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output = AttentionOutput(d_model, dropout, activation_fn)
        if pe: self.proj_p = RotaryPositionalEmbedding(self.d_model)
    
    def forward(self, input_states, memory_states, input_guidance=None, memory_guidance=None, input_coord=None, memory_coord=None):
        """Consistency-Guided Transformer Layer

        Args:
            input_states (Tensor): (B, N, C)
            memory_states (Tensor): (B, M, C)
            input_guidance (Tensor): (B, N, 2)
            memory_guidance (Tensor): (B, M, 2)
            input_coord (Tensor): (B, N, 3)
            memory_coord (Tensor): (B, M, 3)

        Returns:
            output_states: torch.Tensor (B, N, C)
        """
        q = self.proj_q(input_states)
        k = self.proj_k(memory_states)
        v = self.proj_v(memory_states)

        if input_guidance is not None and memory_guidance is not None:
            emb_q1 = self.guidance_emb_q1((input_guidance[:,:,0] / (self.guidance_resolution/5.)).long())
            emb_q2 = self.guidance_emb_q2((input_guidance[:,:,1] / self.guidance_resolution).long())
            q = q + torch.cat([emb_q1, emb_q2], dim=-1)
            
            emb_k1 = self.guidance_emb_k1((memory_guidance[:,:,0] / (self.guidance_resolution/5.)).long())
            emb_k2 = self.guidance_emb_k2((memory_guidance[:,:,1] / self.guidance_resolution).long())
            k = k + torch.cat([emb_k1, emb_k2], dim=-1)

            emb_v1 = self.guidance_emb_v1((memory_guidance[:,:,0] / (self.guidance_resolution/5.)).long())
            emb_v2 = self.guidance_emb_v2((memory_guidance[:,:,1] / self.guidance_resolution).long())
            v = v * torch.cat([emb_v1, emb_v2], dim=-1)
        
        if self.pe and memory_coord is not None and input_coord is not None:
            q = rearrange(self.proj_p(input_coord, q), 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_p(memory_coord, k), 'b m (h c) -> b h m c', h=self.num_heads)
        else:
            q = rearrange(q, 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(k, 'b m (h c) -> b h m c', h=self.num_heads)
        
        v = rearrange(v, 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
    
        hidden_states = self.linear(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        output_states = self.output(output_states)
        return output_states


class GuidedTransformerMoELayer(nn.Module):
    def __init__(self, d_model, num_heads, guidance_bins:int, pe=True, dropout=None, activation_fn='relu'):
        super(GuidedTransformerMoELayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.guidance_resolution = 1. / guidance_bins
        self.pe = pe
        
        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.guidance_emb_q1 = nn.Embedding(guidance_bins*5, self.d_model//2)
        self.guidance_emb_k1 = nn.Embedding(guidance_bins*5, self.d_model//2)
        self.guidance_emb_v1 = nn.Embedding(guidance_bins*5, self.d_model//2)

        self.guidance_emb_q2 = nn.Embedding(guidance_bins, self.d_model//2)
        self.guidance_emb_k2 = nn.Embedding(guidance_bins, self.d_model//2)
        self.guidance_emb_v2 = nn.Embedding(guidance_bins, self.d_model//2)

        nn.init.zeros_(self.guidance_emb_q1.weight)
        nn.init.zeros_(self.guidance_emb_k1.weight)
        nn.init.ones_(self.guidance_emb_v1.weight)

        nn.init.zeros_(self.guidance_emb_q2.weight)
        nn.init.zeros_(self.guidance_emb_k2.weight)
        nn.init.ones_(self.guidance_emb_v2.weight)
        
        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output = GuidedDeepSeekMoE(d_model, 6, guidance_bins=guidance_bins)
        if pe: self.proj_p = RotaryPositionalEmbedding(self.d_model)
    
    def forward(self, input_states, memory_states, input_guidance=None, memory_guidance=None, input_coord=None, memory_coord=None):
        """Consistency-Guided Transformer MoE Layer

        Args:
            input_states (Tensor): (B, N, C)
            memory_states (Tensor): (B, M, C)
            input_guidance (Tensor): (B, N, 2)
            memory_guidance (Tensor): (B, M, 2)
            input_coord (Tensor): (B, N, 3)
            memory_coord (Tensor): (B, M, 3)

        Returns:
            output_states: torch.Tensor (B, N, C)
        """
        q = self.proj_q(input_states)
        k = self.proj_k(memory_states)
        v = self.proj_v(memory_states)

        if input_guidance is not None and memory_guidance is not None:
            emb_q1 = self.guidance_emb_q1((input_guidance[:,:,0] / (self.guidance_resolution/5.)).long())
            emb_q2 = self.guidance_emb_q2((input_guidance[:,:,1] / self.guidance_resolution).long())
            q = q + torch.cat([emb_q1, emb_q2], dim=-1)
            
            emb_k1 = self.guidance_emb_k1((memory_guidance[:,:,0] / (self.guidance_resolution/5.)).long())
            emb_k2 = self.guidance_emb_k2((memory_guidance[:,:,1] / self.guidance_resolution).long())
            k = k + torch.cat([emb_k1, emb_k2], dim=-1)

            emb_v1 = self.guidance_emb_v1((memory_guidance[:,:,0] / (self.guidance_resolution/5.)).long())
            emb_v2 = self.guidance_emb_v2((memory_guidance[:,:,1] / self.guidance_resolution).long())
            v = v * torch.cat([emb_v1, emb_v2], dim=-1)
        
        if self.pe and memory_coord is not None and input_coord is not None:
            q = rearrange(self.proj_p(input_coord, q), 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_p(memory_coord, k), 'b m (h c) -> b h m c', h=self.num_heads)
        else:
            q = rearrange(q, 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(k, 'b m (h c) -> b h m c', h=self.num_heads)
        
        v = rearrange(v, 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
    
        hidden_states = self.linear(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        output_states = self.output(output_states, input_guidance)
        return output_states
