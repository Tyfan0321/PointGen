import torch
import torch.nn as nn
#import torch_scatter
from pytorch3d.ops import knn_gather

import numpy as np
from typing import Dict, List
from scipy.spatial.transform import Rotation as R

from models.kpconv.encoder import KPConvEncoder
from models.cast.ptt import TreeTransformerCrossEncoder, CorrespondenceRegressor
from models.utils import grid_subsample_gpu, radius_search_gpu, apply_transform, weighted_svd


class PTT(nn.Module):
    def __init__(self, cfg):
        super(PTT, self).__init__()
        self.kpconv_layers = cfg.kpconv_layers
        self.voxel_size = cfg.voxel_size
        self.init_radius = cfg.init_radius
        self.neighbor_limits = cfg.neighbor_limits
        self.pyramid_levels = cfg.pyramid_levels
        self.growing_factor = cfg.growing_factor
        self.r_p = self.voxel_size * (2**(self.kpconv_layers-1))
        self.r_n = self.voxel_size * (2**self.kpconv_layers)

        self.backbone = KPConvEncoder(cfg)
        self.proj = nn.Linear(cfg.input_dim_c, cfg.hidden_dim)
        self.transformer = TreeTransformerCrossEncoder(cfg)
        #self.transformer = TransformerCrossEncoder(cfg)
        self.decoder = CorrespondenceRegressor(cfg.hidden_dim)
        self.bce = torch.nn.BCEWithLogitsLoss()

        self.W = torch.nn.Parameter(torch.zeros(cfg.hidden_dim, cfg.hidden_dim), requires_grad=True)
        torch.nn.init.normal_(self.W, std=0.1)
    
    @torch.no_grad()
    def preprocess(self, points, overlap=None):
        if isinstance(points, torch.Tensor):
            points_list = [points]
            length_list = [torch.LongTensor([points.shape[0]]).to(points.device)]
        else:
            points_list = [torch.cat(points)]
            length_list = [torch.LongTensor([p.shape[0] for p in points]).to(points[0].device)]
        
        voxel_size = self.voxel_size
        radius = self.init_radius
        points = points_list[0]
        lengths = length_list[0]

        for _ in range(self.kpconv_layers - 1):
            voxel_size = voxel_size * 2.
            points, lengths = grid_subsample_gpu(points, lengths, voxel_size)
            points_list.append(points); length_list.append(lengths)
        
        neighbors_list = []
        subsampling_list = []
        
        for i in range(self.kpconv_layers):
            neighbors_list.append(radius_search_gpu(
                points_list[i], points_list[i], length_list[i], length_list[i], radius, self.neighbor_limits[i]
            ))
            if i == self.kpconv_layers - 1: break
            subsampling_list.append(radius_search_gpu(
                points_list[i + 1], points_list[i], length_list[i + 1], length_list[i], radius, self.neighbor_limits[i]
            ))
            radius = radius * 2.
        
        if overlap is not None:
            if isinstance(overlap, torch.Tensor):
                overlap_list = [overlap.float()]
            else:
                overlap_list = [torch.cat(overlap, dim=0).float()]
            invalid_indices = [s.sum() for s in length_list]
            for p in range(1, self.kpconv_layers):
                pooling_indices = subsampling_list[p - 1].clone()
                valid_mask = pooling_indices < invalid_indices[p - 1]
                pooling_indices[~valid_mask] = 0
                overlap_gathered = overlap_list[p-1][pooling_indices] * valid_mask
                overlap_gathered = torch.sum(overlap_gathered, dim=1) / torch.sum(valid_mask, dim=1)
                overlap_gathered = torch.clamp(overlap_gathered, min=0, max=1)
                overlap_list.append(overlap_gathered)  # Average pool over indices
            return points_list, neighbors_list, subsampling_list, length_list, overlap_list
        else: return points_list, neighbors_list, subsampling_list, length_list, None
    
    @torch.no_grad()
    def voxelize(self, points: torch.Tensor):
        index_list: List[torch.Tensor] = []
        inverse_list: List[torch.Tensor] = []
        counts_list: List[torch.Tensor] = []
        points_list: List[torch.Tensor] = [points]

        voxel_size = self.voxel_size * (2**(self.kpconv_layers-1))

        for _ in range(1, self.pyramid_levels):
            voxel_size = voxel_size * self.growing_factor
            vi = torch.div(points - points.min(dim=-2,keepdim=True)[0], voxel_size, rounding_mode='floor').long()
            m = (vi[...,:2].max(dim=-2, keepdim=True).values + 1).log2().floor() + 1  # (..., 1, 2)
            cluster = vi[..., 0] + vi[..., 1] * (2**m[...,0]) + vi[..., 2] * (2**(m[...,0]+m[...,1]))  # (..., N)

            _, p2v_map, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True) # (K,),(N,),(K,)
            v2p_map = torch.full([counts.shape[0], counts.max().item()], points.shape[0], dtype=torch.long, device=m.device)  # (K, M)
            mask = torch.arange(v2p_map.shape[-1], dtype=torch.long, device=m.device).unsqueeze(0) < counts.unsqueeze(-1)  # (K, M)
            v2p_map[mask] = torch.argsort(p2v_map)
            
            #points = torch_scatter.scatter_mean(points, p2v_map, dim=0)
            points = torch.cat([points, torch.zeros_like(points[:1])]).unsqueeze(0)  # (1, N+1, 3)
            points = knn_gather(points, v2p_map.unsqueeze(0)).squeeze(0).sum(1)  # (K, 3)
            points = points / counts.float().unsqueeze(-1)  # (K, 3)

            index_list.append(v2p_map)
            inverse_list.append(p2v_map)
            points_list.append(points)
            counts_list.append(counts)

        return {"points": points_list, "index": index_list, "inverse": inverse_list, "counts": counts_list}
    

    def forward(self, ref_points, src_points, ref_overlap=None, src_overlap=None):
        # 1. Preprocess the original point clouds
        points_list, neighbors_list, subsampling_list, length_list, overlap_list = self.preprocess(
            [ref_points[0], src_points[0]], [ref_overlap[0], src_overlap[0]] if ref_overlap is not None else None
        )
        # 2. Extract hierarchical feature maps
        feats = self.proj(self.backbone(points_list, neighbors_list, subsampling_list)[-1])
        ref_points_c = points_list[-1][:length_list[-1][0]]
        src_points_c = points_list[-1][length_list[-1][0]:]
        ref_feats = feats[:length_list[-1][0]]
        src_feats = feats[length_list[-1][0]:]
        
        # 3. Interaction of coarse voxelized features
        ref_feats, src_feats = self.transformer.forward(
            ref_feats, src_feats, self.voxelize(ref_points_c), self.voxelize(src_points_c),
            #ref_feats[None], src_feats[None], ref_points_c[None], src_points_c[None]
        )
        ref_corr, ref_overlap = self.decoder(ref_feats.squeeze(0))
        src_corr, src_overlap = self.decoder(src_feats.squeeze(0))
        pred_pose_weighted = weighted_svd(
            torch.cat([ref_corr, src_points_c]),
            torch.cat([ref_points_c, src_corr]),
            torch.cat([ref_overlap, src_overlap]).squeeze(-1).sigmoid()
        )
        
        outputs = {
            'ref_feat': ref_feats.squeeze(0),
            'src_feat': src_feats.squeeze(0),

            'src_kp': src_points_c,
            'src_kp_warped': src_corr,
            'ref_kp': ref_points_c,
            'ref_kp_warped': ref_corr,

            'src_overlap': src_overlap.squeeze(-1),
            'ref_overlap': ref_overlap.squeeze(-1),

            'pose': pred_pose_weighted,
        }
        if overlap_list is not None:
            outputs['ref_overlap_gt'] = overlap_list[-1][:length_list[-1][0]]
            outputs['src_overlap_gt'] = overlap_list[-1][length_list[-1][0]:]
        
        return outputs

    def compute_similarity(self, ref_feats, src_feats, dual_normalization=False):
        W_triu = torch.triu(self.W)
        W_symmetrical = W_triu + W_triu.T
        match_logits = torch.einsum('...ic,cd,...jd->...ij', ref_feats, W_symmetrical, src_feats)
        if dual_normalization:
            ref_matching_scores = torch.softmax(matching_scores, dim=-1)
            src_matching_scores = torch.softmax(matching_scores, dim=-2)
            matching_scores = ref_matching_scores * src_matching_scores
            return matching_scores
        else: return match_logits

    def compute_infonce(self, match_logits, anchor_xyz, positive_xyz):
        with torch.no_grad():
            dist_keypts = torch.cdist(anchor_xyz, positive_xyz)
            dist1, idx1 = dist_keypts.topk(k=1, dim=-1, largest=False)  # Finds the positive (closest match)
            mask = dist1[..., 0] < self.r_p  # Only consider points with correspondences (..., N_anc)
            ignore = dist_keypts < self.r_n  # Ignore all the points within a certain boundary,
            ignore.scatter_(-1, idx1, 0)     # except the positive (..., N_anc, N_pos)

        match_logits[..., ignore] = -float('inf')
        loss = -torch.gather(match_logits, -1, idx1).squeeze(-1) + torch.logsumexp(match_logits, dim=-1)
        loss = torch.sum(loss * mask.float()) / torch.sum(mask)
        return loss

    def compute_loss(self, output_dict: Dict[str, torch.Tensor]):
        transformed_ref_corr = apply_transform(output_dict['ref_kp_warped'],output_dict['gt_transform'])
        transformed_src_kp = apply_transform(output_dict['src_kp'],output_dict['gt_transform'])
        match_logits = self.compute_similarity(output_dict['ref_feat'], output_dict['src_feat'])
        l_feat = self.compute_infonce(match_logits, output_dict['ref_kp'], transformed_src_kp)

        corr_err = torch.cat([transformed_ref_corr, output_dict['src_kp_warped']], dim=0)
        corr_err = corr_err - torch.cat([output_dict['ref_kp'], transformed_src_kp], dim=0)
        corr_err = torch.sum(torch.abs(corr_err), dim=-1)

        if 'ref_overlap_gt' in output_dict.keys():
            overlap_weights = torch.cat([output_dict['ref_overlap_gt'], output_dict['src_overlap_gt']])
            l_corr = torch.sum(overlap_weights * corr_err) / torch.sum(overlap_weights)
            l_conf = self.bce(torch.cat([output_dict['ref_overlap'], output_dict['src_overlap']]), overlap_weights)
            return l_feat, l_corr, l_conf
        else:
            l_corr = torch.mean(corr_err, dim=1)
            return l_feat, l_corr



class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.rre_threshold = cfg.rre_threshold
        self.rte_threshold = cfg.rte_threshold
        self.inlier_distance_threshold = 0.1
        self.rmse_threshold = 0.2
    
    @torch.no_grad()
    def compute_rmse(self, transform, covariance, estimated_transform):
        relative_transform = torch.matmul(torch.linalg.inv(transform), estimated_transform)
        q = R.from_matrix(relative_transform[:3, :3].cpu().numpy()).as_quat()
        q = torch.from_numpy(q[:3]).float().to(transform.device)
        er = torch.cat([relative_transform[:3, 3], q], dim=-1)
        er = er.view(1, 6) @ covariance @ er.view(6, 1) / covariance[0, 0]
        return torch.sqrt(er)

    @torch.no_grad()
    def transform_error(self, gt_transforms: torch.Tensor, transforms: torch.Tensor):
        rre = 0.5 * ((transforms[:3, :3].T @ gt_transforms[:3, :3]).trace() - 1.0)
        rre = 180.0 * torch.arccos(rre.clamp(-1., 1.)) / np.pi
        rte: torch.Tensor = torch.norm(gt_transforms[:3, 3] - transforms[:3, 3], dim=-1)
        return rte, rre

    @torch.no_grad()
    def forward(self, output_dict: Dict):
        ref_coarse_corr = torch.cat([output_dict['ref_kp'], output_dict['src_kp_warped']])
        src_coarse_corr = torch.cat([output_dict['ref_kp_warped'], output_dict['src_kp']])
        corr_certainty = torch.cat([output_dict['ref_overlap'], output_dict['src_overlap']])
        corr_certainty = corr_certainty / (corr_certainty.max() + 1e-10)
        mask = corr_certainty.gt(0.2)

        ref_coarse_corr = ref_coarse_corr[mask]
        src_coarse_corr = src_coarse_corr[mask]
        corr_certainty = corr_certainty[mask]
        sort_idx = corr_certainty.argsort(descending=True)

        transform = output_dict['gt_transform']
        mask = torch.norm(ref_coarse_corr[sort_idx] - apply_transform(src_coarse_corr[sort_idx], transform), dim=-1)
        mask = torch.lt(mask, self.inlier_distance_threshold).float()
        te, re = self.transform_error(output_dict['gt_transform'], output_dict['pose'])

        results = {
            'TE': te,
            'RE': re,
            'IR': mask.mean(),
            'IR@250': mask[:250].mean(),
            'IR@500': mask[:500].mean(),
        }
        if 'covariance' in output_dict.keys():
            covariance = output_dict['covariance']
            rmse = self.compute_rmse(output_dict['gt_transform'], covariance, output_dict['pose'])
            results['RR'] = rmse.lt(self.rmse_threshold).float()
        else:
            results['RR'] = torch.lt(re, self.rre_threshold) & torch.lt(te, self.rte_threshold)
        
        return results
