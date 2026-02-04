import torch
import torch.nn as nn

from models.utils import grid_subsample_gpu, radius_search_gpu


class PointCloudProcessor:
    """Preprocessor for point cloud data used in REGTR generative model."""
    
    def __init__(
        self,
        kpconv_layers,
        voxel_size,
        init_radius,
        neighbor_limits
    ):
        self.kpconv_layers = kpconv_layers
        self.voxel_size = voxel_size
        self.init_radius = init_radius
        self.neighbor_limits = neighbor_limits
    
    @torch.no_grad()
    def __call__(self, points, overlap=None):
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
            points_list.append(points)
            length_list.append(lengths)
        
        neighbors_list = []
        subsampling_list = []
        
        for i in range(self.kpconv_layers):
            neighbors_list.append(radius_search_gpu(
                points_list[i], points_list[i], length_list[i], length_list[i], radius, self.neighbor_limits[i]
            ))
            if i == self.kpconv_layers - 1: 
                break
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
        else: 
            return points_list, neighbors_list, subsampling_list, length_list, None