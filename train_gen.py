import os
import math
from munch import munchify
import torch
import json
import torch.optim as optim
from data import KittiDataset, IndoorDataset, IndoorTestDataset
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from models.generative.transformer_regtr import RegTrGenerative
from models.generative.processor import PointCloudProcessor
from models.generative.pipeline import Evaluator
from engine.trainer_gen import StepBasedTrainer


class Trainer(StepBasedTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.register_load_save_hooks()
        self.model = RegTrGenerative(**cfg.model, **cfg.loss)
        self.processor = PointCloudProcessor(**cfg.processor)

        if cfg.runname == "kitti":
            train_dataset = KittiDataset(seqs="train", **cfg.data)
            val_dataset = KittiDataset(seqs="val", **cfg.data)
        elif cfg.runname == "3dmatch":
            train_dataset = IndoorDataset(seqs="train", **cfg.data)
            val_dataset = IndoorDataset(seqs="val", **cfg.data)
        else:
            raise KeyError

        self.train_loader = DataLoader(train_dataset, 1, num_workers=cfg.num_workers, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, 1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=cfg.optim.lr, 
            betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2), 
            eps=cfg.optim.adam_eps,
            weight_decay=cfg.optim.adam_weight_decay
            )
        
        # https://github.com/huggingface/diffusers/issues/3954
        print(self.num_train_epochs * math.ceil(len(self.train_loader) / self.gradient_accumulation_steps))
        self.lr_scheduler = get_scheduler(
            cfg.lr_scheduler.name,
            self.optimizer,
            num_training_steps=self.num_train_epochs * math.ceil(len(self.train_loader) / self.gradient_accumulation_steps),
            num_warmup_steps=cfg.lr_scheduler.lr_warmup_steps * self.accelerator.num_processes,
            num_cycles=cfg.lr_scheduler.lr_num_cycles,
            )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(**cfg.scheduler)
        self.evaluator = Evaluator(**cfg.eval)

    def register_load_save_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                # if self.use_ema:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "model"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            # if selfuse_ema:
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = RegTrGenerative.from_pretrained(input_dir, subfolder="model")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./config/kitti_fm.json', type=str)
    parser.add_argument("--resume", type=str, default=None, help=('Use a path saved by `--checkpointing_steps`, or `"latest"`'))
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--allow_tf32", action="store_true")

    _args = parser.parse_args()
    if _args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    
    with open(_args.config, 'r') as f:
        cfg = json.load(f)
        cfg = munchify(cfg)
    
    Trainer(
        cfg=cfg,
        weighting_scheme=_args.weighting_scheme,
        logit_mean=_args.logit_mean,
        logit_std=_args.logit_std,
    ).fit(_args.resume)