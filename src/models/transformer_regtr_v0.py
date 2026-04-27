import math
from tkinter import NONE
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils.outputs import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNorm, AdaLayerNormZeroSingle
# from diffusers.models.attention_dispatch import dispatch_attention_fn



@dataclass
class RegTrModelOutput(BaseOutput):
    sample: "torch.Tensor"
    overlap_gt: "torch.Tensor" = None
    extra_loss: Dict[str, "torch.Tensor"] = None
    # overlap_pred: Optional["torch.Tensor"]

class TimestepProjEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=emb_dim)

    def forward(self, timesteps, hidden_states):
        timesteps_proj = self.time_proj(timesteps)
        timesteps_emb = self.time_embedder(timesteps_proj.to(dtype=hidden_states.dtype))

        return timesteps_emb

class RegTrPositioinEmbedding(nn.Module):
    """
    Args:
        in_channels: Number of input channles, e.g. 2 for image coordinates, 3 for pc coordinates
        emb_dim: Number of dimensions to encode into
        temperature:
        scale:
    """
    def __init__(self, in_channels: int = 3, emb_dim: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.in_channels = in_channels
        self.per_channel_dim = emb_dim // in_channels // 2 * 2
        self.temperature = temperature
        self.padding = emb_dim - self.per_channel_dim * self.in_channels

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_channels

        half_dim = self.per_channel_dim // 2
        exponent = -math.log(self.temperature) * torch.arange(0, half_dim, dtype=torch.float32, device=x.device) 
        exponent = exponent / half_dim
        emb = torch.exp(exponent) # (*, d_out/d_in/2)

        x = x * self.scale
        emb = x.unsqueeze(-1) * emb # (*, d_in, d_out/d_in/2)

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # (*, d_in, d_out/d_in)

        emb = emb.reshape(*x.shape[:-1], -1) # (*, d_out)

        # Pad unused dimensions with zeros
        emb = F.pad(emb, (0, self.padding))
        return emb


class RegTrGeometricEmedding(nn.Module):
    def __init__(self, emb_dim: int, sigma_d: float, sigma_a: float, angle_k: int, reduction_a: str = "max"):
        super().__init__()
        self.emb_dim = emb_dim
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.angle_k = angle_k
        self.reduction_a = reduction_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, 3] point coordinates

        Returns:
            emb: [N, d_model] embedding
        """
        # Compute pairwise distances
        distances = torch.cdist(x, x)  # [N, N]
        
        # Compute distance-based embeddings
        dist_emb = torch.exp(-distances ** 2 / (2 * self.sigma_d ** 2))  # [N, N]
        
        # For angle-based embeddings, we need to compute angles between point pairs
        # This is a simplified version - in a full implementation, you might compute
        # local geometric features like normals or curvature
        N = x.shape[0]
        angle_emb = torch.zeros(N, N, device=x.device)
        
        # Combine embeddings
        if self.reduction_a == "max":
            emb = torch.max(dist_emb, angle_emb)
        else:
            emb = dist_emb * angle_emb
            
        # Project to d_model dimensions
        emb = emb.unsqueeze(-1).repeat(1, 1, self.d_model)
        return emb


class RegTrGenerative(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        # Encoder configuration
        encoder_config: dict,
        # RegDiT specific parameters
        # # Geometric structure embedding parameters
        # sigma_d: float = 0.2,
        # sigma_a: float = 15,
        # angle_k: int = 3,
        # reduction_a: str = "max",
        in_dim: int = 3,
        out_dim: Optional[int] = None,
        hidden_dim: int = 256,
        num_attention_heads: int = 8,
        attention_head_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        activation_fn: str = "geglu",
        norm_type: str = "layer_norm", # "layer_norm", "ada_norm", "ada_norm_zero"
        num_layers: int = 6,
        ## Loss parameters
        learn_w: bool = True,
        r_p: Optional[float] = None,
        r_n: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        self.encoder_type = encoder_config.get("type", "")
        self.encoder = self._init_encoder(encoder_config)       
        self.in_dim_context = self.encoder.out_channels

        self.pos_emb = RegTrPositioinEmbedding(in_channels=3, emb_dim=hidden_dim)
        self.time_emb = TimestepProjEmbedding(emb_dim=hidden_dim)
        self.scale_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.proj_in = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.extra = 1 if norm_type == "layer_norm" else 0
        self.denoise_transformer = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=hidden_dim // num_attention_heads if attention_head_dim is None else attention_head_dim,
                dropout=dropout if dropout is not None else 0.0,
                cross_attention_dim=hidden_dim, # For cross-attention
                activation_fn=activation_fn,
                norm_type=norm_type,
            )
            for _ in range(num_layers)
        ])

        # Context transformer (no timesteps conditioning but with standard normalization and cross-attention)
        self.proj_in_context = nn.Linear(self.in_dim_context, hidden_dim)
        self.context_transformer = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=hidden_dim // num_attention_heads if attention_head_dim is None else attention_head_dim,
                dropout=dropout if dropout is not None else 0.0,
                cross_attention_dim=hidden_dim,  # For cross-attention
                activation_fn=activation_fn,
                norm_type="layer_norm",
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        out_dim = out_dim if out_dim is not None else in_dim
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)
        self.norm_out_context = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)
        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim, bias=False)
        )

        
        # Extra Loss
        self.r_p = r_p
        self.r_n = r_n

        # self.ov_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, 1)
        # )
        self.ov_head = None

        self.scale = kwargs.get("scale") or [1, 1, 1]
    
    def _init_encoder(self, encoder_config):
        """
        Initialize encoder based on configuration
        
        Args:
            encoder_config: Encoder configuration dictionary
        
        Returns:
            Initialized encoder module
        """        
        if self.encoder_type == "kpconv":
            from src.models.encoders import KPConvEncoder
            return KPConvEncoder(
                encoder_config.get("kpconv_layers", 4),
                encoder_config.get("input_dim", 1),
                encoder_config.get("kernel_size", 15),
                encoder_config.get("init_dim", 64),
                encoder_config.get("init_sigma", 0.05),
                encoder_config.get("init_radius", 0.0625)
            )
        elif self.encoder_type == "sonata":
            from src.models.encoders import SonataEncoder
            return SonataEncoder(
                encoder_config.get("pretrained", True),
                encoder_config.get("freeze", True),
                encoder_config.get("pretrained_ckpt", None),
                encoder_config.get("layer_index", 0)
            )
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def forward(
        self,
        sample: torch.FloatTensor,
        timesteps: Union[torch.Tensor, float, int],
        ref_points_c: torch.Tensor,
        src_points_c: torch.Tensor,
        tgt_points_c: torch.Tensor,
        encoder_inputs: List,
        overlap_list: Optional[List[torch.FloatTensor]] = None,
        tgt_points_c_corr: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        Args:
            sample (`torch.FloatTensor`): (B, N, C) noisy inputs tensor
            timesteps (`torch.FloatTensor` or `float` or `int`): (B,) timesteps
            ref_points_c (`torch.FloatTensor`): (N, 3) reference points
            src_points_c (`torch.FloatTensor`): (M, 3) source points
            encoder_inputs (`Union[Dict[str, Any], List[torch.FloatTensor]]`): Encoder inputs provided by model_processor.py
            overlap_list (`List[torch.FloatTensor]`): List of overlap
            tgt_points_c (`torch.FloatTensor`): (M, 3) target points, used for infonce loss
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`RegTrModelOutput`] instead of a plain tuple.
        """
        if self.encoder_type == "kpconv":
            points_list, neighbors_list, subsampling_list, length_list = encoder_inputs
            ref_feats_origin, src_feats_origin = self.encoder(points_list, neighbors_list, subsampling_list, length_list)
        elif self.encoder_type == "sonata":
            ref_point, src_point = encoder_inputs
            
            ref_layers = self.encoder(ref_point)
            src_layers = self.encoder(src_point)
            layer_index = self.encoder.layer_index
            ref_feats_origin = ref_layers[layer_index]["feat"]
            src_feats_origin = src_layers[layer_index]["feat"]

            
            assert ref_feats_origin.shape[0] == ref_points_c.shape[0]
            assert src_feats_origin.shape[0] == src_points_c.shape[0]
        else:
            raise ValueError(f"Unsupported Encoder Type{self.encoder_type}")

        if overlap_list is not None:            
            ref_ov_gt, src_ov_gt = overlap_list[-1][:ref_points_c.shape[0]], overlap_list[-1][ref_points_c.shape[0]:]
        else:
            ref_ov_gt, src_ov_gt = None, None
        
        ref_points_c = ref_points_c.unsqueeze(0)
        src_points_c = src_points_c.unsqueeze(0)
        tgt_points_c = tgt_points_c.unsqueeze(0)
        ref_feats_origin = ref_feats_origin.unsqueeze(0)
        src_feats_origin = src_feats_origin.unsqueeze(0)

        ref_feats = self.proj_in_context(ref_feats_origin)
        src_feats = self.proj_in_context(src_feats_origin)

        ref_pos_emb = self.pos_emb(ref_points_c)
        src_pos_emb = self.pos_emb(src_points_c)

        for layer_context in self.context_transformer:
            ref_feats_new = layer_context(ref_feats, None, src_feats, ref_pos_emb, src_pos_emb)
            src_feats_new = layer_context(src_feats, None, ref_feats, src_pos_emb, ref_pos_emb)
            ref_feats = ref_feats_new
            src_feats = src_feats_new
        
        # sample_pos_emb = sample * (torch.std(ref_points_c, dim=1) + 1e-8) + torch.mean(ref_points_c, dim=1)
        # sample_pos_emb = self.pos_emb(sample_pos_emb)
        ref_feats_b = ref_feats.expand(sample.shape[0], -1, -1)
        src_feats_b = src_feats.expand(sample.shape[0], -1, -1)

        ref_points_c_b = ref_points_c.expand(sample.shape[0], -1, -1)
        ref_feats_b = torch.cat([ref_points_c_b, ref_feats_b], dim=-1)
        ref_feats_b = self.proj_in(ref_feats_b)

        hidden_states = torch.cat([sample, src_feats_b], dim=-1)
        hidden_states = self.proj_in(hidden_states)
        temb = self.time_emb(timesteps, hidden_states)

        if scale is not None:
            # We expect scale to be a batched scalar
            log_scale = torch.log(scale + 1e-8)
            if log_scale.dim() == 1:
                log_scale = log_scale.unsqueeze(-1)
            temb = temb + self.scale_emb(log_scale.to(dtype=hidden_states.dtype))

        if self.extra > 0:
            hidden_states = torch.cat([temb[:, None], hidden_states], dim=1)
            temb = None

        for layer in self.denoise_transformer:
            hidden_states = layer(hidden_states, temb, ref_feats_b, src_pos_emb, ref_pos_emb)

        if self.extra > 0:
            hidden_states = hidden_states[:, self.extra:]

        sample = self.norm_out(hidden_states)
        sample = self.proj_out(sample)

        ref_feats = self.norm_out_context(ref_feats)
        src_feats = self.norm_out_context(src_feats)
        
        if self.ov_head:
            ref_ov_pred = self.ov_head(ref_feats).squeeze(-1)  # (1, N)
            src_ov_pred = self.ov_head(src_feats).squeeze(-1)  # (1, M)
        else:
            ref_ov_pred, src_ov_pred = None, None
        
        # extra_loss = self.compute_extra_loss(
        #     ref_feats_origin, src_feats_origin, ref_points_c, tgt_points_c,
        #     ref_ov_pred, src_ov_pred, ref_ov_gt, src_ov_gt
        # )
        extra_loss = {}
        if not return_dict:
            return (sample, src_ov_gt, extra_loss)
            
        return RegTrModelOutput(
            sample=sample, 
            overlap_gt=src_ov_gt,
            extra_loss=extra_loss,
            # overlap_pred=src_ov_pred,
        )
    
    def compute_extra_loss(
        self, 
        ref_feats, 
        src_feats, 
        ref_points, 
        tgt_points, 
        ref_ov_pred=None,
        src_ov_pred=None,
        ref_ov_gt=None,
        src_ov_gt=None,
        dual_normalization=False,
    ):
        encoder_frozen = False
        try:
            encoder_frozen = all(not param.requires_grad for param in self.encoder.parameters())
        except:
            pass
        
        if encoder_frozen:
            with torch.no_grad():
                return self._compute_extra_loss_impl(
                    ref_feats, src_feats, ref_points, tgt_points, 
                    ref_ov_pred, src_ov_pred, ref_ov_gt, src_ov_gt, 
                    dual_normalization
                )
        else:
            return self._compute_extra_loss_impl(
                ref_feats, src_feats, ref_points, tgt_points, 
                ref_ov_pred, src_ov_pred, ref_ov_gt, src_ov_gt, 
                dual_normalization
            )
    
    def _compute_extra_loss_impl(
        self, 
        ref_feats, 
        src_feats, 
        ref_points, 
        tgt_points, 
        ref_ov_pred=None,
        src_ov_pred=None,
        ref_ov_gt=None,
        src_ov_gt=None,
        dual_normalization=False,
    ):
        ref_feats = F.normalize(ref_feats, p=2, dim=-1)
        src_feats = F.normalize(src_feats, p=2, dim=-1)

        match_logits = torch.einsum('...ic,...jc->...ij', ref_feats, src_feats)
        match_logits /= 0.1 
        with torch.no_grad():
            dist_keypts = torch.cdist(ref_points, tgt_points)
            dist1, idx1 = dist_keypts.topk(k=1, dim=-1, largest=False)  # Finds the positive (closest match)
            mask = dist1[..., 0] < self.r_p  # Only consider points with correspondences (..., N_anc)
            ignore = dist_keypts < self.r_n  # Ignore all the points within a certain boundary,
            ignore.scatter_(-1, idx1, 0)     # except the positive (..., N_anc, N_pos)

        match_logits[..., ignore] = -float('inf')

        if dual_normalization:
            ref_match_logits = torch.softmax(match_logits, dim=-1)
            src_match_logits = torch.softmax(match_logits, dim=-2)
            match_logits = ref_match_logits * src_match_logits
        
        infonce_loss = -torch.gather(match_logits, -1, idx1).squeeze(-1) + torch.logsumexp(match_logits, dim=-1)
        infonce_loss = torch.sum(infonce_loss * mask.float()) / (torch.sum(mask) + 1e-5)

        if ref_ov_pred is not None and src_ov_pred is not None and ref_ov_gt is not None and src_ov_gt is not None:
            ref_bce = F.binary_cross_entropy_with_logits(ref_ov_pred.squeeze(0), ref_ov_gt.float())
            src_bce = F.binary_cross_entropy_with_logits(src_ov_pred.squeeze(0), src_ov_gt.float())
            bce_loss = (ref_bce + src_bce) / 2.0
        else:
            bce_loss = None

        return {'infonce_loss': infonce_loss, 'bce_loss': bce_loss}
    

class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        use_additive_timestep_norm (`bool`, *optional*, defaults to `False`): 
            Whether to use additive timesteps normalization (AdaLN) or standard normalization.
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_type: str = "layer_norm", # "layer_norm", "ada_norm", "ada_norm_zero"
        attention_bias: bool = False,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
            )
        else:
            self.attn2 = None

        self.norm_type = norm_type
        if norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim)
            self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
            self.norm3 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
            self.norm3 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)

        if self.attn2 is not None:
            if norm_type == "ada_norm_zero":
                self.norm2 = AdaLayerNormZeroSingle(dim)
            elif norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim)
            else:
                self.norm2 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)

            self.norm2_encoder = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        else:
            self.norm2 = None
            self.norm2_encoder = None

    def with_pos_embed(self, feat, pos: Optional[torch.Tensor]):
        return feat if pos is None else feat + pos

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        pos_emb: Optional[torch.FloatTensor] = None,
        encoder_pos_emb: Optional[torch.FloatTensor] = None
    ):
        if temb is not None:
            if self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp= self.norm1(hidden_states, emb=temb)
            elif self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, emb=temb)
            else:
                norm_hidden_states = self.norm1(hidden_states)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        norm_hidden_states = self.with_pos_embed(norm_hidden_states, pos_emb)
        attn_output = self.attn1(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa[:, None] * attn_output
        hidden_states = attn_output + hidden_states

        if self.attn2 and self.norm2 and self.norm2_encoder and encoder_hidden_states is not None:
            if temb is not None and self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_mca = self.norm2(hidden_states, emb=temb)
            elif temb is not None and self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, emb=temb)
            else:
                norm_hidden_states = self.norm2(hidden_states)
            norm_encoder_hidden_states = self.norm2_encoder(encoder_hidden_states)

            norm_hidden_states = self.with_pos_embed(norm_hidden_states, pos_emb)
            norm_encoder_hidden_states = self.with_pos_embed(norm_encoder_hidden_states, encoder_pos_emb)
            attn_output = self.attn2(norm_hidden_states, norm_encoder_hidden_states)

            if self.norm_type == "ada_norm_zero":
                attn_output = gate_mca[:, None] * attn_output   
            hidden_states = attn_output + hidden_states

        if temb is not None and self.norm_type == "ada_norm_zero":
            norm_hidden_states = self.norm3(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        elif temb is not None and self.norm_type == "ada_norm":
            norm_hidden_states = self.norm3(hidden_states, emb=temb)
        else:
            norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp[:, None] * ff_output
        hidden_states = ff_output + hidden_states

        return hidden_states

