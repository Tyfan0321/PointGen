import os
import datetime
import math
import logging
from tqdm import tqdm
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
import diffusers
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from engine.summary_board import SummaryBoard
from models.utils import apply_transform, weighted_svd
from models.generative.pipeline import PointGenPipeline

from munch import unmunchify 
import json


class StepBasedTrainer:
    def __init__(self, cfg, **kwargs):
        if cfg.seed is not None:
            set_seed(cfg.seed)
        self.output_dir = cfg.output_dir
        self.logging_dir = os.path.join(self.output_dir, cfg.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.output_dir, logging_dir=self.logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with=cfg.log_with,
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f"config.json"), 'w', encoding='utf-8') as f_out:
                json.dump(unmunchify(cfg), f_out, indent=2, ensure_ascii=False)

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        if torch.backends.mps.is_available():
            # Disable AMP for MPS.
            self.accelerator.native_amp = False
            if self.weight_dtype == torch.bfloat16 or self.weight_dtype == torch.float16:
                raise ValueError("Mixed precision training is not supported on MPS. Please use fp32 instead.")        

        # logging setting
        self.logger = get_logger("Accelerate Trainer")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            diffusers.utils.logging.set_verbosity_info()
        else:
            diffusers.utils.logging.set_verbosity_error()

        # training entities
        self.noise_scheduler: None
        self.model: None
        self.processor: None
        self.optimizer: None
        self.lr_scheduler: None
        self.evaluator: None
        self.loss_func: None

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        # training variables
        self.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self.train_batch_size = cfg.train_batch_size
        self.num_train_epochs = cfg.num_train_epochs
        self.num_train_steps = cfg.num_train_steps
        self.log_steps = cfg.log_steps
        self.ckpt_epochs = cfg.ckpt_epochs
        self.clip_grad = cfg.clip_grad_norm
        self.feat_stop_epoch = cfg.feat_stop_epoch
        self.scale = cfg.model.scale if cfg.model.scale else [1, 1, 1]

        # evaluation entities
        self.val_epochs = cfg.val_epochs
        self.do_gen = cfg.do_gen
        if self.do_gen:
            self.num_gen_samples = cfg.num_gen_samples
            self.num_inference_steps = cfg.num_inference_steps
            self.inference_type = cfg.inference_type
        self.summary_board = SummaryBoard(last_n=self.log_steps, adaptive=True)

        # fm advance settings
        self.weighting_scheme = kwargs.get("weighting_scheme", None)
        self.logit_mean = kwargs.get("logit_mean", None)
        self.logit_std = kwargs.get("logit_std", None)
        self.mode_scale = kwargs.get("mode_scale", None)

    def get_sigmas(self, timesteps, n_dim=3, dtype=torch.float32):
        ## For Flow Matching
        sigmas = self.noise_scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def prepare_noisy_data(self, data_dict, do_normalize=True):
        ref_points = data_dict.get("ref_points")
        src_points = data_dict.get("src_points")
        ref_overlap = data_dict.get("ref_overlap", None)
        src_overlap = data_dict.get("src_overlap", None)
        Tr = data_dict.get("Tr", None).squeeze(0)
        points_list, neighbors_list, subsampling_list, length_list, overlap_list = self.processor(
            [ref_points[0], src_points[0]], [ref_overlap[0], src_overlap[0]] if ref_overlap is not None else None
        )

        ref_points_c = points_list[-1][:length_list[-1][0]].to(dtype=self.weight_dtype)
        src_points_c = points_list[-1][length_list[-1][0]:].to(dtype=self.weight_dtype)

        tgt_points_c = src_points_c.clone()
        tgt_points_c = apply_transform(tgt_points_c, Tr)
        target = tgt_points_c.unsqueeze(0).expand(self.train_batch_size, *tgt_points_c.shape)

        # dist_keypts = torch.cdist(ref_points_c, tgt_points_c)
        # dist_min, closest_ref_indices = torch.min(dist_keypts, dim=0)
        # tgt_points_c_corr = ref_points_c[closest_ref_indices]
        # target = (tgt_points_c - tgt_points_c_corr).unsqueeze(0).expand(self.train_batch_size, *tgt_points_c.shape)

        noise = torch.randn_like(target, device=target.device)

        if do_normalize:
            target = (target - torch.mean(ref_points_c, dim=0)) / (torch.std(ref_points_c, dim=0) + 1e-8)
            # target = target / target.new_tensor(self.scale)
        
        ## for diffsuion
        # timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (self.train_batch_size,), device=target.device).long()
        # sample = self.noise_scheduler.add_noise(target, noise, timesteps)

        ## for flow matching
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=self.train_batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=target.device)

        sigmas = self.get_sigmas(timesteps, n_dim=target.ndim, dtype=target.dtype)
        sample = sigmas * noise + (1.0 - sigmas) * target

        train_data_dict = {
            "sample": sample, 
            "noise": noise,
            "target": target,
            "timesteps": timesteps, 
            "ref_points_c": ref_points_c,
            "src_points_c": src_points_c,
            "tgt_points_c": tgt_points_c,
            # "tgt_points_c_corr": tgt_points_c_corr,
            "points_list": points_list,
            "neighbors_list": neighbors_list,
            "subsampling_list": subsampling_list,
            "overlap_list": overlap_list,
            "sigmas": sigmas
        }
        return train_data_dict
    
    def get_extra_loss_weight(self, current_epoch):
        """Calculate the dynamic weight for extra_loss based on cosine decay."""
        base = 1.0
        if not self.feat_stop_epoch:
            return base
        if current_epoch < self.feat_stop_epoch:
            weight = 0.5 * (1 + math.cos(math.pi * current_epoch / self.feat_stop_epoch))
        else:
            weight = 0.0
        return weight * base
    
    def val_step(self, data_dict):
        noise = data_dict.pop("noise")
        target = data_dict.pop("target")
        sigmas = data_dict.pop("sigmas")
        sample = data_dict.get("sample")

        model_ouput = self.model(**data_dict)

        # if self.noise_scheduler.config.prediction_type == "epsilon":
        #     loss_target = noise
        # elif self.noise_scheduler.config.prediction_type == "sample":
        #     loss_target = target
        # else:
        #     raise ValueError(
        #         f"Unsupported prediction_type: {self.noise_scheduler.config.prediction_type}"
        #     )
        # model_ouput = self.model(**data_dict)
        # loss = F.mse_loss(model_ouput.sample.float(), loss_target.float(), reduction="mean")

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        target_std_weight = torch.std(data_dict["ref_points_c"], dim=0)
        target_std_norm = torch.norm(target_std_weight).item()

        v = (sample - target) / sigmas.clamp_min(5e-2)
        x_pred = model_ouput.sample
        v_pred = (sample - x_pred) / sigmas.clamp_min(5e-2)

        loss = torch.mean(
            (weighting.float() * (v_pred.float() - v.float()) ** 2).reshape(target.shape[0], -1),
            dim=1,
        ).mean()

        # model_pred = model_ouput.sample * (-sigmas) + data_dict["sample"]
        # loss = torch.mean(
        #     (weighting.float() * ((model_pred.float() - target.float()) * target_std_weight) ** 2).reshape(target.shape[0], -1),
        #     dim=1,
        # ).mean()

        infonce_loss = model_ouput.extra_loss["infonce_loss"]
        bce_loss = model_ouput.extra_loss["bce_loss"]
        if bce_loss:
            extra_loss = infonce_loss + bce_loss
        else:
            extra_loss = infonce_loss

        return {
            "loss": loss.detach().item(),
            "infonce_loss": infonce_loss.detach().item(),
            # "bce_loss": bce_loss.detach().item() if bce_loss else None,
            # "extra_loss": extra_loss.detach().item(),
            "target_std_norm": target_std_norm,
        }

    def step(self, data_dict, current_epoch=0) -> Dict[str,torch.Tensor]:
        noise = data_dict.pop("noise")
        target = data_dict.pop("target")
        sigmas = data_dict.pop("sigmas")
        sample = data_dict.get("sample")

        model_ouput = self.model(**data_dict)

        # if self.noise_scheduler.config.prediction_type == "epsilon":
        #     loss_target = noise
        # elif self.noise_scheduler.config.prediction_type == "sample":
        #     loss_target = target
        # else:
        #     raise ValueError(
        #         f"Unsupported prediction_type: {self.noise_scheduler.config.prediction_type}"
        #     )
        # loss = F.mse_loss(model_ouput.sample.float(), loss_target.float(), reduction="mean")

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        target_std_weight = torch.std(data_dict["ref_points_c"], dim=0)
        target_std_norm = torch.norm(target_std_weight).item()

        v = (target - sample) / sigmas.clamp_min(5e-2)
        x_pred = model_ouput.sample
        v_pred = (x_pred - sample) / sigmas.clamp_min(5e-2)

        loss = torch.mean(
            (weighting.float() * (v_pred.float() - v.float()) ** 2).reshape(target.shape[0], -1),
            dim=1,
        ).mean()


        # model_pred = model_ouput.sample * (-sigmas) + data_dict["sample"]
        # loss = torch.mean(
        #     (weighting.float() * ((model_pred.float() - target.float()) * target_std_weight) ** 2).reshape(target.shape[0], -1),
        #     dim=1,
        # ).mean()

        infonce_loss = model_ouput.extra_loss["infonce_loss"]
        bce_loss = model_ouput.extra_loss["bce_loss"]
        if bce_loss:
            extra_loss = infonce_loss + bce_loss
        else:
            extra_loss = infonce_loss

        # Calculate dynamic weight for extra_loss
        # extra_loss_weight = self.get_extra_loss_weight(current_epoch)
        # overall_loss = loss + extra_loss_weight * extra_loss.float()
        if current_epoch < self.feat_stop_epoch:
            overall_loss = loss + extra_loss.float()
        else:
            overall_loss = loss

        self.accelerator.backward(overall_loss)
        if self.accelerator.sync_gradients:
            params_to_clip = self.model.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, self.clip_grad)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return {
            "loss": loss.detach().item(),
            "infonce_loss": infonce_loss.detach().item(),
            # "bce_loss": bce_loss.detach().item() if bce_loss else None,
            # "extra_loss": extra_loss.detach().item(),
            # "overall_loss": overall_loss.detach().item(),
            "target_std_norm": target_std_norm
        }

    def val_gen(self, mode="train"):
        if mode == "train":
            dataloader = self.train_loader
        elif mode == "val":
            dataloader = self.val_loader
        elif mode == "test":
            dataloader = self.test_loader
        else:
            assert "Unsupported mode."

        transformer = self.accelerator.unwrap_model(self.model)
        pipeline = PointGenPipeline(
            scheduler=self.noise_scheduler,
            processor=self.processor,
            transformer=transformer,
            scheduler_type=self.inference_type,
        )
        generator = torch.Generator(device=self.accelerator.device).manual_seed(0)

        dist_error = []
        dist_ov_errror = []
        all_re = []
        all_te = []
        all_rr = []
        all_re_ov = []
        all_te_ov = []
        all_rr_ov = []
        overlap_precision = []
        overlap_recall = []

        progress_bar = tqdm(dataloader, desc="Test", disable=not self.accelerator.is_local_main_process)

        for i, data_dict in enumerate(progress_bar):
            data_dict = {k: v.to(self.accelerator.device) for k, v in data_dict.items()}
            
            pred_points, tgt_points, ref_points, tgt_points_corr, src_points, gt_overlap = pipeline(data_dict, num_inference_steps=self.num_inference_steps, generator=generator)  
            pred_points = pred_points.squeeze(0)
            pred_points = pred_points * torch.std(ref_points, dim=0) + torch.mean(ref_points, dim=0)
            # pred_points = pred_points * pred_points.new_tensor(self.scale) + tgt_points_corr
            per_point_dist_error = torch.norm(pred_points - tgt_points, dim=-1).mean()
            dist_error.append(per_point_dist_error)

            overlap_mask = gt_overlap > 0.5
            if overlap_mask.sum() > 0:
                ov_dist_error = torch.norm(pred_points[overlap_mask] - tgt_points[overlap_mask], dim=-1).mean()
                dist_ov_errror.append(ov_dist_error)
            
                # if pred_overlap is not None:
                #     pred_overlap = pred_overlap.squeeze(0)
                #     pred_overlap = pred_overlap > 0.5
                #     precision = (pred_overlap[overlap_mask].sum()) / pred_overlap.sum()
                #     recall = (pred_overlap[overlap_mask].sum()) / overlap_mask.sum()
                #     overlap_precision.append(precision)
                #     overlap_recall.append(recall)
            pred_transform = weighted_svd(src_points.squeeze(0), pred_points)
            te, re, rr = self.evaluator(data_dict["Tr"].squeeze(0), pred_transform)
            all_te.append(te.float().item())
            all_re.append(re.float().item())
            all_rr.append(rr.float().item())


            pred_transform_ov = weighted_svd(src_points.squeeze(0)[overlap_mask], pred_points[overlap_mask])
            te_ov, re_ov, rr_ov = self.evaluator(data_dict["Tr"].squeeze(0), pred_transform_ov)
            all_te_ov.append(te_ov.float().item())
            all_re_ov.append(re_ov.float().item())
            all_rr_ov.append(rr_ov.float().item())

            logs = {
                "RE": re.item(), "TE": te.item(), "RR": rr.item(),
                "REO": re_ov.item(), "TEO": te_ov.item(), "RRO": rr_ov.item(),
                }
            progress_bar.set_postfix(**logs)
            
            if i == self.num_gen_samples - 1 and mode != "test":
                break

        val_gen_log = {
            "dist_error": torch.tensor(dist_error).mean().detach().item(),
            "dist_ov_errror": torch.tensor(dist_ov_errror).mean().detach().item(),
            # "overlap_precision": torch.tensor(overlap_precision).mean().detach().item(),
            # "overlap_recall": torch.tensor(overlap_recall).mean().detach().item(),
            "RRE": torch.tensor(all_re).mean().item(),
            "RTE": torch.tensor(all_te).mean().item(),
            "RR": torch.tensor(all_rr).mean().item(),
            "RREO": torch.tensor(all_re_ov).mean().item(),
            "RTEO": torch.tensor(all_te_ov).mean().item(),
            "RRO": torch.tensor(all_rr_ov).mean().item()
        }

        return {mode + "_" + k: v for k, v in val_gen_log.items()}

    def log_epoch_dict(self, epoch, log_dict):
        msg = f"Epoch {epoch + 1}-"
        for k, v in log_dict.items():
            msg += f"{k}: {v:.4f}, "
        self.logger.info(msg)
        self.accelerator.log(log_dict, step=epoch + 1)

    def fit(self, resume_from_checkpoint=None):
        now = datetime.datetime.now()
        project_name = now.strftime("%Y-%m-%d_%H-%M-%S")
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(project_name)

        self.model, self.optimizer, self.lr_scheduler, ddp_train_loader, ddp_val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_loader, self.val_loader
        )
        num_steps_per_epoch = math.ceil(len(ddp_train_loader) / self.gradient_accumulation_steps)
        self.num_train_steps = num_steps_per_epoch * self.num_train_epochs
        total_batch_size = self.train_batch_size * self.accelerator.num_processes * self.gradient_accumulation_steps

        if resume_from_checkpoint:
            # Resume training
            if resume_from_checkpoint != "latest":
                path = os.path.basename(resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                resume_from_checkpoint = None
                global_step = 0
                resume_epoch = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.output_dir, path))
                resume_epoch = int(path.split("-")[-1]) + 1
                global_step = resume_epoch * num_steps_per_epoch
        else:
            global_step = 0
            resume_epoch = 0

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num batches each epoch = {len(ddp_train_loader)}")
        self.logger.info(f"  Num Epochs = {self.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.num_train_steps}")

        progress_bar = tqdm(
            range(0, self.num_train_steps),
            initial=global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )
        std = torch.zeros(3, device=self.accelerator.device)
        mean = torch.zeros(3, device=self.accelerator.device)
        for epoch in range(resume_epoch, self.num_train_epochs):
            self.model.train()
            for step, data_dict in enumerate(ddp_train_loader): 
                train_data_dict = self.prepare_noisy_data(data_dict)
                with self.accelerator.accumulate(self.model):
                    loss_dict = self.step(train_data_dict, current_epoch=epoch)

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                logs = {**loss_dict, "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                if global_step % self.log_steps == 0 and self.accelerator.is_main_process:
                    self.accelerator.log(logs, step=global_step)

            self.accelerator.wait_for_everyone()

            if (epoch + 1) % self.ckpt_epochs == 0 or epoch == self.num_train_epochs - 1:
                if self.accelerator.is_main_process:
                    save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
                    self.accelerator.save_state(save_path)
                    self.logger.info(f"Saved state to {save_path}")


            if (epoch + 1) % self.val_epochs == 0 or epoch == self.num_train_epochs - 1:
                self.model.eval()
                total_loss = torch.tensor(0.0, device=self.accelerator.device)
                total_infonce_loss = torch.tensor(0.0, device=self.accelerator.device)
                # total_extra_loss = torch.tensor(0.0, device=self.accelerator.device)
                # total_bce_loss = torch.tensor(0.0, device=self.accelerator.device)
                num_samples = torch.tensor(0, device=self.accelerator.device)
                for step, data_dict in enumerate(ddp_val_loader):
                    with torch.no_grad():
                        val_data_dict = self.prepare_noisy_data(data_dict)
                        val_loss_dict = self.val_step(val_data_dict)
                    total_loss += val_loss_dict["loss"]
                    total_infonce_loss += val_loss_dict["infonce_loss"]
                    # total_extra_loss += val_loss_dict["extra_loss"]
                    # total_bce_loss += val_loss_dict["bce_loss"]
                    num_samples += 1
                gathered_losses = self.accelerator.gather(total_loss)
                gathered_infonce_loss = self.accelerator.gather(total_infonce_loss)
                # gathered_extra_losses = self.accelerator.gather(total_extra_loss)
                # gathered_bce_loss = self.accelerator.gather(total_bce_loss)
                gathered_samples = self.accelerator.gather(num_samples)

                if self.accelerator.is_main_process: 
                    global_loss = gathered_losses.sum() / gathered_samples.sum()
                    global_infonce_loss = gathered_infonce_loss.sum() / gathered_samples.sum()
                    # global_bce_loss = gathered_bce_loss.sum() / gathered_samples.sum()
                    # global_extra_loss = gathered_extra_losses.sum() / gathered_samples.sum()
                    val_log = {
                        "val_loss": global_loss.item(),
                        "val_infonce_loss": global_infonce_loss.item(),
                        # "val_bce_loss": global_bce_loss.item(),
                        # "val_extra_loss": global_extra_loss.item(),
                    }
                    self.log_epoch_dict(epoch, val_log)
                    if self.do_gen:
                        train_gen_log = self.val_gen("train")
                        val_gen_log = self.val_gen("val")
                        gen_log = {**train_gen_log, **val_gen_log}
                        self.log_epoch_dict(epoch, gen_log)
            self.accelerator.wait_for_everyone()

        self.accelerator.end_training()

    def eval(self, load_from_checkpoint):
        self.model = self.accelerator.prepare(self.model)
        if load_from_checkpoint != "latest" and load_from_checkpoint:
            path = os.path.basename(load_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(self.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
            path = dirs[-1] if len(dirs) > 0 else None

        self.accelerator.print(f"Loading from checkpoint {path}")
        self.accelerator.load_state(os.path.join(self.output_dir, path))

        self.logger.info("***** Running evaluation *****")
        msg = f"\n Checkpoint {os.path.join(self.output_dir, path)} Steps {self.num_inference_steps}:"
        self.logger.info(msg)
        val_gen_log = self.val_gen("test")
        for k, v in val_gen_log.items():
            msg += f"{k}: {v:.4f}, "
        self.logger.info(msg)
        if self.accelerator.is_main_process:
            log_file_path = os.path.join(self.output_dir, "eval_results.txt")
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


if __name__ == "__main__":
    import json
    from munch import munchify

    with open("./config/kitti_fm.json", 'r') as f:
        cfg = munchify(json.load(f))

    trainer = StepBasedTrainer(cfg)            



                    
