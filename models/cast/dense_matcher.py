import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.kpconv.kpconv import KPConv, DepthwiseKPConv
from models.kpconv.modules import ResidualBlock


class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x:torch.Tensor, y:torch.Tensor, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1,keepdim=True) * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = torch.exp((c - 1.0) / T)
        return K

class GaussianProcessRegression(nn.Module):
    def __init__(self, T=1, learn_temperature=False, gp_dim=64, basis="fourier", sigma_noise=0.1):
        super().__init__()
        self.K = CosKernel(T, learn_temperature)
        self.sigma_noise = sigma_noise
        self.pos_emb = nn.Linear(3, gp_dim)
        self.basis = basis
        self.dim = gp_dim

    def get_pos_enc(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_emb(x))
        else: return self.pos_emb(x)

    def forward(self, src_feats, tgt_feats, tgt_points):
        f = self.get_pos_enc(tgt_points)
        K_yy = self.K(tgt_feats, tgt_feats)
        K_xy = self.K(src_feats, tgt_feats)
        sigma_noise = self.sigma_noise * torch.eye(K_yy.shape[-1], device=f.device)[None, :, :]
        # Due to https://github.com/pytorch/pytorch/issues/16963 annoying warnings, remove batch if N large
        K_yy_inv = torch.inverse(K_yy + sigma_noise)
        return K_xy.matmul(K_yy_inv.matmul(f))


class ChannalAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannalAttentionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):  # high, low (old, new)
        x = torch.cat([x1, x2], dim=-1).mean(1, keepdim=True)  # (B, N, C) -> (B, 1, C)
        return self.conv(x) * x2 + x1

class DiscriminativeFeatureNet(nn.Module):  # 512, 256, 256, 384, 15, ?,?
    def __init__(self, input_feat_dim, emb_dim, feat_dim, dfn_dim, kernel_size, radius, sigma):
        super().__init__()
        self.feat_input_proj = nn.Linear(input_feat_dim, feat_dim)
        self.rrb_d = ResidualBlock(emb_dim + feat_dim, dfn_dim, kernel_size, radius, sigma)
        self.cab = ChannalAttentionBlock(2 * dfn_dim, dfn_dim)
        self.rrb_u = ResidualBlock(dfn_dim, dfn_dim, kernel_size, radius, sigma)
        self.head = nn.Linear(dfn_dim, 4)

    def forward(self, embeddings, feats, context, points, neighbor_indices):
        feats = self.feat_input_proj(feats)
        embeddings = torch.cat([feats, embeddings], dim=1)
        embeddings = self.rrb_d(embeddings, points, points, neighbor_indices)
        context = self.cab([context, embeddings])
        context = self.rrb_u(context, points, points, neighbor_indices)
        preds = self.head(context)
        pred_coord = preds[:, -3:]
        pred_certainty = preds[:, :-3]
        return pred_coord, pred_certainty, context


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=True,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = True
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.disp_emb = nn.Conv2d(3,displacement_emb_dim,1,1,0)
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
    
    def create_block(self, in_dim, out_dim, dw=False, kernel_size=5):
        num_groups = 1 if not dw else in_dim
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
        )
        norm = nn.BatchNorm2d(out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, x, y, flow):
        """Computes the relative refined displacement in pixels for a given image x,y and a coarse flow-field between them

        Args:
            x ([type]): [description]
            y ([type]): [description]
            flow ([type]): [description]

        Returns:
            [type]: [description]
        """
        device = x.device
        b,c,hs,ws = x.shape
        with torch.no_grad():
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False)
        if self.has_displacement_emb:
            query_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            )
            )
            query_coords = torch.stack((query_coords[1], query_coords[0]))
            query_coords = query_coords[None].expand(b, 2, hs, ws)
            in_displacement = flow-query_coords
            emb_in_displacement = self.disp_emb(in_displacement)
            if self.local_corr_radius:
                if self.corr_in_other:
                    # Corr in other means take a kxk grid around the predicted coordinate in other image
                    local_corr = local_correlation(x,y,local_radius=self.local_corr_radius,flow=flow)
                else:
                    # Otherwise we use the warp to sample in the first image
                    # This is actually different operations, especially for large viewpoint changes
                    local_corr = local_correlation(x,x_hat,local_radius=self.local_corr_radius)
                if self.no_support_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
            else:
                d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
        else:
            if self.no_support_fm:
                x_hat = torch.zeros_like(x)
            d = torch.cat((x, x_hat), dim=1)
        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d)
        certainty, displacement = d[:, :-2], d[:, -2:]
        return certainty, displacement
