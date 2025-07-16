import torch
import torch.nn as nn
from pytorch3d.ops import knn_points
from models.kpconv.backbone import KPConvFPN
from models.cast.cast_dm import SpotGuidedGeoTransformer_mlp as SpotGuidedGeoTransformer
from models.utils import grid_subsample_gpu, radius_search_gpu, apply_transform, weighted_svd
from models.cast.consistency import registration_ransac_based_on_correspondence


class CAST(nn.Module):
    def __init__(self, cfg):
        super(CAST, self).__init__()
        self.kpconv_layers = cfg.kpconv_layers
        self.voxel_size = cfg.voxel_size
        self.init_radius = cfg.init_radius
        self.neighbor_limits = cfg.neighbor_limits

        self.sigma_r = cfg.sigma_r
        self.ransac_filter = cfg.ransac_filter

        self.backbone = KPConvFPN(cfg)
        self.transformer = SpotGuidedGeoTransformer(cfg)
    
    @torch.no_grad()
    def preprocess(self, points):
        points_list, length_list = [points], []
        lengths = torch.LongTensor([points.shape[0]]).to(points.device)
        length_list.append(lengths)

        voxel_size = self.voxel_size
        radius = self.init_radius

        for _ in range(self.kpconv_layers - 1):
            voxel_size = voxel_size * 2.
            points, lengths = grid_subsample_gpu(points, lengths, voxel_size)
            points_list.append(points); length_list.append(lengths)
        
        neighbors_list = []
        subsampling_list = []
        upsampling_list = [None]
        
        for i in range(self.kpconv_layers):
            neighbors_list.append(radius_search_gpu(
                points_list[i], points_list[i], length_list[i], length_list[i], radius, self.neighbor_limits[i]
            ))
            if i == self.kpconv_layers - 1: break
            subsampling_list.append(radius_search_gpu(
                points_list[i + 1], points_list[i], length_list[i + 1], length_list[i], radius, self.neighbor_limits[i]
            ))
            radius = radius * 2.
            if i == 0: continue
            upsampling_list.append(torch.squeeze(knn_points(
                points_list[i].unsqueeze(0), points_list[i + 1].unsqueeze(0))[1], dim=0)
            )
        return points_list, neighbors_list, subsampling_list, upsampling_list
    
    def forward(self, ref_points, src_points, gt_transform):
        # 1. Preprocess the original point clouds
        points_list1, neighbors_list1, subsampling_list1, upsampling_list1 = self.preprocess(ref_points[0])
        points_list2, neighbors_list2, subsampling_list2, upsampling_list2 = self.preprocess(src_points[0])

        # 2. Extract hierarchical feature maps and sparse keypoints
        ref_feats = self.backbone(points_list1, neighbors_list1, subsampling_list1, upsampling_list1)
        src_feats = self.backbone(points_list2, neighbors_list2, subsampling_list2, upsampling_list2)
        output_dict = {'gt_transform': gt_transform[0]}
        
        # 3. Interaction of coarse voxelized features
        ref_corr, src_corr, correlation = self.transformer(
            points_list1[2].unsqueeze(0),
            points_list2[2].unsqueeze(0),
            ref_feats[1].unsqueeze(0),
            src_feats[1].unsqueeze(0),
            points_list1[-1].unsqueeze(0),
            points_list2[-1].unsqueeze(0),
            ref_feats[-1].unsqueeze(0),
            src_feats[-1].unsqueeze(0),
        )
        ref_corr_certainty = torch.sigmoid(ref_corr[..., :1]).squeeze() # (M,)
        src_corr_certainty = torch.sigmoid(src_corr[..., :1]).squeeze() # (N,)

        ref_corr = ref_corr[0, :, 1:] # (M, 3)
        src_corr = src_corr[0, :, 1:] # (N, 3)

        output_dict['spot_matching_scores'] = torch.cat(correlation)
        output_dict['ref_coarse_corr'] = (points_list1[2], ref_corr)
        output_dict['src_coarse_corr'] = (points_list2[2], src_corr)
        output_dict['ref_coarse_certainty'] = ref_corr_certainty
        output_dict['src_coarse_certainty'] = src_corr_certainty
        
        with torch.no_grad():
            # 4. Generate ground-truth patch correspondences
            dist = torch.cdist(points_list1[2], apply_transform(points_list2[2], gt_transform[0]))
            dist = torch.clamp_max(dist / self.sigma_r, 2.)
            overlap = torch.relu(1. + dist.pow(3) / 16. - 0.75 * dist)
            output_dict['gt_patch_overlap'] = overlap
        
        if not self.training:
            corr_certainty = torch.cat([ref_corr_certainty, src_corr_certainty])
            ref_corr,src_corr = torch.cat([points_list1[2], src_corr]), torch.cat([ref_corr, points_list2[2]])
            transform = registration_ransac_based_on_correspondence(ref_corr, src_corr, corr_certainty, topk=250)
            mask = torch.norm(apply_transform(src_corr, transform) - ref_corr, dim=-1).lt(self.ransac_filter)
            if mask.int().sum() < 4:
                output_dict['transform'] = torch.eye(4, device=mask.device)
            else:
                transform = weighted_svd(src_corr, ref_corr, corr_certainty.masked_fill(~mask, 0.))
            output_dict['transform'] = transform
        
        return output_dict