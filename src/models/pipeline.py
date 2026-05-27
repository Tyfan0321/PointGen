import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import numpy as np
from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler, 
    FlowMatchEulerDiscreteScheduler
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor

from src.utils.point_cloud_utils import apply_transform


SCHEDULER_MAP = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
    "dpm": DPMSolverMultistepScheduler,
    "fm-euler": FlowMatchEulerDiscreteScheduler,
}

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class PointGenPipeline(DiffusionPipeline):
    def __init__(
        self, 
        scheduler,
        processor,
        transformer,
        scheduler_type: str = "ddim",
    ):
        super().__init__() 
        if scheduler_type in SCHEDULER_MAP:
            scheduler = SCHEDULER_MAP[scheduler_type].from_config(scheduler.config)

        self.register_modules(
            scheduler=scheduler,
            processor=processor,
            transformer=transformer,
        )

    def prepare_data(self, data_dict):
        ref_points = data_dict.get("ref_points")
        src_points = data_dict.get("src_points")
        ref_overlap = data_dict.get("ref_overlap", None)
        src_overlap = data_dict.get("src_overlap", None)

        model_type = next(self.transformer.parameters()).dtype

        processor_output = self.processor(
            [ref_points[0], src_points[0]], [ref_overlap[0], src_overlap[0]] if ref_overlap is not None else None
        )
        processor_type = self.processor.type
        if processor_type == "kpconv":
            points_list, neighbors_list, subsampling_list, length_list, overlap_list = processor_output
            ref_points_c = points_list[-1][:length_list[-1][0]].to(dtype=model_type)
            src_points_c = points_list[-1][length_list[-1][0]:].to(dtype=model_type)
            encoder_inputs = (points_list, neighbors_list, subsampling_list)
        elif processor_type == "sonata":
            points_list, overlap_list = processor_output
            ref_point_dict = points_list[0]
            src_point_dict = points_list[1]
            layer_index = getattr(self.processor, "layer_index", 0)
            if "pooling_cache" in ref_point_dict and "pyramid" in ref_point_dict["pooling_cache"]:
                pyramid_ref = list(reversed(ref_point_dict["pooling_cache"]["pyramid"]))
                pyramid_src = list(reversed(src_point_dict["pooling_cache"]["pyramid"]))
                layer_index = min(layer_index, len(pyramid_ref) - 1)
                ref_points_c = pyramid_ref[layer_index]["coord"].to(dtype=model_type)
                src_points_c = pyramid_src[layer_index]["coord"].to(dtype=model_type)
            else:
                ref_points_c = ref_point_dict["coord"].to(dtype=model_type)
                src_points_c = src_point_dict["coord"].to(dtype=model_type)
            
            ref_center_shift = ref_point_dict.get("center_shift", torch.zeros(3, device=ref_points_c.device))
            src_center_shift = src_point_dict.get("center_shift", torch.zeros(3, device=src_points_c.device))
            ref_points_c = ref_points_c + ref_center_shift
            src_points_c = src_points_c + src_center_shift
            
            encoder_inputs = [ref_point_dict, src_point_dict]

        Tr = data_dict.get("Tr")
        tgt_points_c = apply_transform(src_points_c, Tr[0])
        dist_keypts = torch.cdist(ref_points_c, tgt_points_c)
        dist_min, closest_ref_indices = torch.min(dist_keypts, dim=0)
        tgt_points_c_corr = ref_points_c[closest_ref_indices]
        
        scale = torch.std(ref_points_c) + 1e-8

        return {
            "ref_points_c": ref_points_c,
            "src_points_c": src_points_c,
            "tgt_points_c": tgt_points_c,
            "tgt_points_c_corr": tgt_points_c_corr,
            "encoder_inputs": encoder_inputs,
            "overlap_list": overlap_list,
            "scale": scale.unsqueeze(0),
        }
    
    def prepare_sample(self, batch_size, num_channels, length, dtype, device, generator):
        shape = (batch_size, length, num_channels)
        sample = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return sample

    @torch.no_grad()
    def __call__(
        self,
        data_dict,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        device = self._execution_device
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device)
        sigmas = self.scheduler.sigmas

        model_data_dict = self.prepare_data(data_dict)
        length, num_channels = model_data_dict["src_points_c"].shape

        model_type = next(self.transformer.parameters()).dtype

        sample = self.prepare_sample(
            batch_size=1,
            num_channels=num_channels,
            length=length,
            dtype=model_type,
            device=device,
            generator=generator
        )
        # sample = model_data_dict["src_points_c"].unsqueeze(0)

        for t, s in zip(timesteps, sigmas):
            v_pred, ov_gt = self.transformer(sample, t.unsqueeze(0), **model_data_dict, return_dict=False)[:2]
            # x_pred, ov_gt = self.transformer(sample, t.unsqueeze(0), **model_data_dict, return_dict=False)[:2]
            # v_pred = (x_pred - sample ) / s

            # if t == timesteps[0]:
            #     x_pred_gt = (model_data_dict["tgt_points_c"] - torch.mean(model_data_dict["ref_points_c"], dim=0)) / (torch.std(model_data_dict["ref_points_c"], dim=0) + 1e-8)
            #     v_pred_gt = (x_pred_gt - sample) / s
            #     v_pred = v_pred_gt
            sample = self.scheduler.step(-v_pred, t, sample).prev_sample
        return (sample, model_data_dict["tgt_points_c"], model_data_dict["ref_points_c"], model_data_dict["tgt_points_c_corr"],  model_data_dict["src_points_c"], ov_gt)
    

class Evaluator(nn.Module):
    def __init__(self, rre_threshold, rte_threshold):
        super(Evaluator, self).__init__()
        self.rre_threshold = rre_threshold
        self.rte_threshold = rte_threshold
        self.inlier_distance_threshold = 0.1

    @torch.no_grad()
    def transform_error(self, gt_transforms: torch.Tensor, transforms: torch.Tensor):
        rre = 0.5 * ((transforms[:3, :3].T @ gt_transforms[:3, :3]).trace() - 1.0)
        rre = 180.0 * torch.arccos(rre.clamp(-1., 1.)) / np.pi
        rte: torch.Tensor = torch.norm(gt_transforms[:3, 3] - transforms[:3, 3], dim=-1)
        return rte, rre

    @torch.no_grad()
    def forward(self, gt_transform, pred_transform):
        te, re = self.transform_error(gt_transform, pred_transform)
        rr = torch.lt(re, self.rre_threshold) & torch.lt(te, self.rte_threshold)
        
        return te, re, rr
