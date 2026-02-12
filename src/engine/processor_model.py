import torch
import torch.nn as nn

from src.utils.point_cloud_utils import grid_subsample_gpu, radius_search_gpu
from src.models.sonata import transform as sonata_transform
from src.models.sonata.utils import offset2batch
import torch_scatter


class KPConvPointCloudProcessor:
    """Preprocessor for point cloud data used in REGTR generative model with KPConv backbone."""
    
    def __init__(
        self,
        kpconv_layers,
        voxel_size,
        init_radius,
        neighbor_limits
    ):
        self.type = "kpconv"
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


class SonataPointCloudProcessor:
    """Preprocessor for point cloud data used in REGTR generative model with Sonata backbone."""
    def __init__(self, type, stride=(2, 2, 2, 2), build_pooling_cache=True, layer_index=0):
        self.type = type
        self.stride = stride
        self.build_pooling_cache = build_pooling_cache
        self.layer_index = layer_index
        self.transform = sonata_transform.default()

    def _to_device(self, value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {k: self._to_device(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_device(v, device) for v in value]
        return value

    @torch.no_grad()
    def __call__(self, points, overlap=None):
        f"""
        Args: 
            points: list[torch.Tensor], each [N, 3]
        """
        if isinstance(points, torch.Tensor):
            points = [points]

        if overlap is not None:
            if isinstance(overlap, torch.Tensor):
                overlap_list = [overlap.float()]
            else:
                overlap_list = [torch.cat(overlap, dim=0).float()]
        
        points_list = []
        device = points[0].device
        for point_cloud in points:
            point = {
                "coord": point_cloud,
                "color": point_cloud.new_zeros(point_cloud.shape[0], 3),
                "normal": point_cloud.new_zeros(point_cloud.shape[0], 3),
            }
            point_numpy = {}
            for key, value in point.items():
                if isinstance(value, torch.Tensor):
                    point_numpy[key] = value.cpu().numpy()
                else:
                    point_numpy[key] = value

            sonata_point = self.transform(point_numpy)
            if self.build_pooling_cache:
                sonata_point["pooling_cache"] = self._build_pooling_cache(sonata_point)
            points_list.append(sonata_point)
        
        points_list = self._to_device(points_list, device)
        
        return points_list, overlap_list

    def _build_pooling_cache(self, point):
        grid_coord = point["grid_coord"]
        coord = point["coord"]
        batch = point.get("batch", None)
        if batch is None:
            batch = offset2batch(point["offset"])

        stages = []
        pyramid = [{"coord": coord, "grid_coord": grid_coord, "batch": batch}]

        for stride in self.stride:
            grid_coord_down = torch.div(grid_coord, stride, rounding_mode="trunc")
            grid_key = grid_coord_down | (batch.view(-1, 1) << 48)
            unique, cluster, counts = torch.unique(
                grid_key,
                sorted=True,
                return_inverse=True,
                return_counts=True,
                dim=0,
            )
            indices = torch.argsort(cluster)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            head_indices = indices[idx_ptr[:-1]]

            coord_down = torch_scatter.segment_csr(coord[indices], idx_ptr, reduce="mean")
            batch_down = batch[head_indices]
            grid_coord_unique = unique & ((1 << 48) - 1)

            stages.append(
                {
                    "stride": stride,
                    "grid_coord": grid_coord_unique,
                    "cluster": cluster,
                    "indices": indices,
                    "idx_ptr": idx_ptr,
                    "head_indices": head_indices,
                }
            )
            pyramid.append({"coord": coord_down, "grid_coord": grid_coord_unique, "batch": batch_down})

            coord = coord_down
            grid_coord = grid_coord_unique
            batch = batch_down

        return {"stages": stages, "pyramid": pyramid, "cursor": 0}


def create_point_cloud_processor(processor_type, **kwargs):
    if processor_type == "kpconv":
        return KPConvPointCloudProcessor(**kwargs)
    elif processor_type == "sonata":
        return SonataPointCloudProcessor(**kwargs)
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}")
