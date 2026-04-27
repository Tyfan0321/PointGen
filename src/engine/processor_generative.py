import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from src.utils.point_cloud_utils import apply_transform


class DiffusionDataProcessor:
    def __init__(self, cfg, processor=None, noise_scheduler=None):
        self.train_batch_size = cfg.train_batch_size
        self.weighting_scheme = cfg.diffusion.weighting_scheme
        self.logit_mean = cfg.diffusion.logit_mean
        self.logit_std = cfg.diffusion.logit_std
        self.mode_scale = cfg.diffusion.mode_scale

        self.processor = processor
        self.noise_scheduler = noise_scheduler
    
    
    def prepare_noisy_data(self, data_dict):
        ref_points = data_dict.get("ref_points")
        src_points = data_dict.get("src_points")
        ref_overlap = data_dict.get("ref_overlap", None)
        src_overlap = data_dict.get("src_overlap", None)
        Tr = data_dict.get("Tr", None).squeeze(0)
        
        if self.processor.type == "kpconv":
            processor_output = self.processor(
                [ref_points[0], src_points[0]], 
                [ref_overlap[0], src_overlap[0]] if ref_overlap is not None else None
            )
            points_list, neighbors_list, subsampling_list, length_list, overlap_list = processor_output
            ref_points_c = points_list[-1][:length_list[-1][0]]
            src_points_c = points_list[-1][length_list[-1][0]:]
            encoder_inputs = [points_list, neighbors_list, subsampling_list, length_list]

        elif self.processor.type == "sonata":
            points_list, overlap_list = self.processor(
                [ref_points[0], src_points[0]], 
                [ref_overlap[0], src_overlap[0]] if ref_overlap is not None else None
            )
            ref_point_dict = points_list[0]
            src_point_dict = points_list[1]

            layer_index = getattr(self.processor, "layer_index", 0)
            if "pooling_cache" in ref_point_dict and "pyramid" in ref_point_dict["pooling_cache"]:
                pyramid_ref = list(reversed(ref_point_dict["pooling_cache"]["pyramid"]))
                pyramid_src = list(reversed(src_point_dict["pooling_cache"]["pyramid"]))
                layer_index = min(layer_index, len(pyramid_ref) - 1)
                ref_points_c = pyramid_ref[layer_index]["coord"]
                src_points_c = pyramid_src[layer_index]["coord"]
            else:
                ref_points_c = ref_point_dict["coord"]
                src_points_c = src_point_dict["coord"]

            ref_center_shift = ref_point_dict.get("center_shift", torch.zeros(3, device=ref_points_c.device))
            src_center_shift = src_point_dict.get("center_shift", torch.zeros(3, device=src_points_c.device))
            ref_points_c = ref_points_c + ref_center_shift
            src_points_c = src_points_c + src_center_shift

            encoder_inputs = [ref_point_dict, src_point_dict]

        else:
            raise ValueError(f"Unsupported Processor Type{self.processor.type}")

        
        tgt_points_c = src_points_c.clone()
        tgt_points_c = apply_transform(tgt_points_c, Tr)
        target = tgt_points_c.unsqueeze(0).expand(self.train_batch_size, *tgt_points_c.shape)
        
        noise = torch.randn_like(target, device=target.device)
        
        target = (target - torch.mean(ref_points_c, dim=0)) / (torch.std(ref_points_c) + 1e-8)
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=self.train_batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        indices = indices.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
        timesteps = self.noise_scheduler.timesteps[indices].to(device=target.device)
        
        sigmas = self.get_sigmas(timesteps, target.ndim, target.dtype)
        
        sample = sigmas * noise + (1.0 - sigmas) * target
        
        return {
            "sample": sample,
            "timesteps": timesteps,
            "ref_points_c": ref_points_c,
            "src_points_c": src_points_c,
            "tgt_points_c": tgt_points_c,
            "encoder_inputs": encoder_inputs,
            "overlap_list": overlap_list,
            "noise": noise,
            "target": target,
            "sigmas": sigmas
        }
    
    def get_sigmas(self, timesteps, n_dim, dtype):
        sigmas = self.noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(timesteps.device)
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
        
        v = (target - sample) / sigmas.clamp_min(5e-5)
        v_pred = model_output.sample
        # x_pred = model_output.sample
        # v_pred = (x_pred - sample) / sigmas.clamp_min(5e-5)
        
        loss = torch.mean(
            (weighting.float() * (v_pred.float() - v.float()) ** 2).reshape(target.shape[0], -1),
            dim=1,
        ).mean()

        # loss = torch.mean(
        #     (weighting.float() * (x_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        #     dim=1,
        # ).mean()
        

        return {
            "loss": loss.detach().item(),
            "overall_loss": loss
        }
        
        # infonce_loss = model_output.extra_loss["infonce_loss"]
        # bce_loss = model_output.extra_loss["bce_loss"]
        
        # if bce_loss:
        #     extra_loss = infonce_loss + bce_loss
        # else:
        #     extra_loss = infonce_loss
        
        # if current_epoch < feat_stop_epoch:
        #     overall_loss = loss + extra_loss.float()
        # else:
        #     overall_loss = loss
        
        # return {
        #     "loss": loss.detach().item(),
        #     "infonce_loss": infonce_loss.detach().item(),
        #     "overall_loss": overall_loss
        # }
