import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.transformer.vanilla_transformer import MultiHeadAttention
from models.transformer.rpe_transformer import RPEMultiHeadAttention
from models.transformer.positional_encoding import GeometricStructureEmbedding


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



class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, activation="relu"):
        super().__init__()

        # Self, cross attention layers
        self.sa = RPEMultiHeadAttention(d_model, nhead)
        self.ca = MultiHeadAttention(d_model, nhead)
        #self.linear_sa = nn.Linear(d_model, d_model)
        #self.linear_ca = nn.Linear(d_model, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, tgt, src_pos, tgt_pos, src_rpe_emb, tgt_rpe_emb):
        # Self attention
        src2 = self.norm1(src) + src_pos
        src2,_ = self.sa(src2, src2, src2, src_rpe_emb)
        src = src + src2 #self.linear_sa(src2)  # 
        
        tgt2 = self.norm1(tgt) + tgt_pos
        tgt2,_ = self.sa(tgt2, tgt2, tgt2, tgt_rpe_emb)
        tgt = tgt + tgt2 #self.linear_sa(tgt2)  # 

        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        src3,_ = self.ca(src_w_pos, tgt_w_pos, tgt_w_pos)
        tgt3,_ = self.ca(tgt_w_pos, src_w_pos, src_w_pos)

        src = src + src3 #self.linear_ca(src3)  # 
        tgt = tgt + tgt3 #self.linear_ca(tgt3)  # 

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.activation(self.linear1(src2)))
        src = src + src2

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        tgt = tgt + tgt2

        return src, tgt



class TransformerCrossEncoder(nn.Module):
    def __init__(self, cfg, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerCrossEncoderLayer(cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim) for _ in range(cfg.blocks)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.return_intermediate = return_intermediate

        self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.hidden_dim)
        self.embed = GeometricStructureEmbedding(
            cfg.hidden_dim, cfg.sigma_d, cfg.sigma_a, cfg.angle_k, cfg.reduction_a
        )

    def forward(self, src, tgt, src_pos, tgt_pos):
        src_pos_emb = self.pos_embed(src_pos)
        tgt_pos_emb = self.pos_embed(tgt_pos)
        ref_embeddings = self.embed(src_pos)
        src_embeddings = self.embed(tgt_pos)

        src_intermediate, tgt_intermediate = [], []

        for layer in self.layers:
            src, tgt = layer(src, tgt, src_pos_emb, tgt_pos_emb, ref_embeddings, src_embeddings)
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
