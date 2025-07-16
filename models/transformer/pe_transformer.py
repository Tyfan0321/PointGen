import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.transformer.positional_encoding import RotaryPositionalEmbedding, SinusoidalPositionalEmbedding
from models.transformer.output_layer import AttentionOutput


class PEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, pe='rope'):
        super(PEMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.pe = pe # ['none', 'rope', 'sine', 'linear', 'fourier']

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        if self.pe == 'rope':
            self.proj_p = RotaryPositionalEmbedding(self.d_model)
        elif self.pe == 'linear' or self.pe == 'fourier':
            self.proj_p = nn.Linear(3, self.d_model)
        elif self.pe == 'sine':
            self.proj_p = SinusoidalPositionalEmbedding(int(math.ceil(self.d_model/3)))

        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)

    def forward(self, input_q, input_k, input_v, embed_q, embed_k, key_masks=None, attention_factors=None):
        """Self-attention with positional embedding forward propagation.

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_q: torch.Tensor (B, N, 3)
            embed_k: torch.Tensor (B, M, 3)
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        if self.pe == 'rope':
            q = rearrange(self.proj_p(embed_q, self.proj_q(input_q)), 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_p(embed_k, self.proj_k(input_k)), 'b m (h c) -> b h m c', h=self.num_heads)
        elif self.pe == 'linear':
            q = rearrange(self.proj_q(input_q) + self.proj_p(embed_q), 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_k) + self.proj_p(embed_k), 'b m (h c) -> b h m c', h=self.num_heads)
        elif self.pe == 'fourier':
            q = rearrange(self.proj_q(input_q) + torch.cos(8*math.pi*self.proj_p(embed_q)), 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_k) + torch.cos(8*math.pi*self.proj_p(embed_k)), 'b m (h c) -> b h m c', h=self.num_heads)
        elif self.pe == 'sine':
            embed_q = rearrange(self.proj_p(embed_q), 'b n c d -> b n (c d)')[:,:,:self.d_model]
            embed_k = rearrange(self.proj_p(embed_k), 'b m c d -> b m (c d)')[:,:,:self.d_model]
            q = rearrange(self.proj_q(input_q) + embed_q, 'b n (h c) -> b h n c', h=self.num_heads)
            k = rearrange(self.proj_k(input_k) + embed_k, 'b m (h c) -> b h m c', h=self.num_heads)

        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        return hidden_states, attention_scores


class PEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, pe='rope'):
        super(PEAttentionLayer, self).__init__()
        self.attention = PEMultiHeadAttention(d_model, num_heads, dropout=dropout, pe=pe)
        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        input_embeddings,
        memory_embeddings,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            input_embeddings,
            memory_embeddings,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class PETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='relu', pe='rope'):
        super(PETransformerLayer, self).__init__()
        self.attention = PEAttentionLayer(d_model, num_heads, dropout=dropout, pe=pe)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        input_embeddings,
        memory_embeddings,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            input_embeddings,
            memory_embeddings,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores
