import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


class DiffusionDataProcessor:
    def __init__(self, cfg):
        self.train_batch_size = cfg.train_batch_size
        self.weighting_scheme = cfg.diffusion.weighting_scheme
        self.logit_mean = cfg.diffusion.logit_mean
        self.logit_std = cfg.diffusion.logit_std
        self.processor_config = cfg.processor
    
    def prepare_noisy_data(self, data_dict, noise_scheduler):
        ref_points = data_dict.get("ref_points")
        src_points = data_dict.get("src_points")
        ref_overlap = data_dict.get("ref_overlap", None)
        src_overlap = data_dict.get("src_overlap", None)
        Tr = data_dict.get("Tr", None).squeeze(0)
        
        from src.engine.model_processor import PointCloudProcessor
        processor = PointCloudProcessor(**self.processor_config)
        points_list, neighbors_list, subsampling_list, length_list, overlap_list = processor(
            [ref_points[0], src_points[0]], 
            [ref_overlap[0], src_overlap[0]] if ref_overlap is not None else None
        )
        
        ref_points_c = points_list[-1][:length_list[-1][0]]
        src_points_c = points_list[-1][length_list[-1][0]:]
        
        tgt_points_c = src_points_c.clone()
        from src.utils.point_cloud_utils import apply_transform
        tgt_points_c = apply_transform(tgt_points_c, Tr)
        target = tgt_points_c.unsqueeze(0).expand(self.train_batch_size, *tgt_points_c.shape)
        
        noise = torch.randn_like(target, device=target.device)
        
        target = (target - torch.mean(ref_points_c, dim=0)) / (torch.std(ref_points_c, dim=0) + 1e-8)
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=self.train_batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=target.device)
        
        sigmas = self.get_sigmas(timesteps, noise_scheduler, target.ndim, target.dtype)
        
        sample = sigmas * noise + (1.0 - sigmas) * target
        
        return {
            "sample": sample,
            "timesteps": timesteps,
            "ref_points_c": ref_points_c,
            "src_points_c": src_points_c,
            "tgt_points_c": tgt_points_c,
            "points_list": points_list,
            "neighbors_list": neighbors_list,
            "subsampling_list": subsampling_list,
            "overlap_list": overlap_list,
            "noise": noise,
            "target": target,
            "sigmas": sigmas
        }
    
    def get_sigmas(self, timesteps, noise_scheduler, n_dim, dtype):
        sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(timesteps.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def compute_loss(self, model_output, data_dict, current_epoch, feat_stop_epoch):
        target = data_dict["target"]
        sample = data_dict["sample"]
        sigmas = data_dict["sigmas"]
        
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme, 
            sigmas=sigmas
        )
        
        v = (target - sample) / sigmas.clamp_min(5e-2)
        x_pred = model_output.sample
        v_pred = (x_pred - sample) / sigmas.clamp_min(5e-2)
        
        loss = torch.mean(
            (weighting.float() * (v_pred.float() - v.float()) ** 2).reshape(target.shape[0], -1),
            dim=1,
        ).mean()
        
        infonce_loss = model_output.extra_loss["infonce_loss"]
        bce_loss = model_output.extra_loss["bce_loss"]
        
        if bce_loss:
            extra_loss = infonce_loss + bce_loss
        else:
            extra_loss = infonce_loss
        
        if current_epoch < feat_stop_epoch:
            overall_loss = loss + extra_loss.float()
        else:
            overall_loss = loss
        
        return {
            "loss": loss.detach().item(),
            "infonce_loss": infonce_loss.detach().item(),
            "overall_loss": overall_loss
        }
