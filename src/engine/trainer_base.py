import os
import datetime
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from tqdm import tqdm

from src.utils.config_utils import save_config_snapshot


class BaseTrainer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.logging_dir = os.path.join(self.output_dir, cfg.log_with)
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        if cfg.seed is not None:
            set_seed(cfg.seed)
        
        accelerator_project_config = ProjectConfiguration(project_dir=self.output_dir, logging_dir=self.logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with=cfg.log_with,
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        
        self.logger = get_logger("BaseTrainer")
        
        self.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self.train_batch_size = cfg.train_batch_size
        self.num_train_epochs = cfg.num_train_epochs
        self.log_steps = cfg.log_steps
        self.ckpt_epochs = cfg.ckpt_epochs
        self.val_epochs = cfg.val_epochs
        self.clip_grad_norm = cfg.clip_grad_norm
        
        self.do_gen = cfg.do_gen
        if self.do_gen:
            self.num_gen_samples = cfg.num_gen_samples
            self.num_inference_steps = cfg.num_inference_steps
            self.inference_type = cfg.inference_type
        
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
    
    @abstractmethod
    def prepare_data(self):
        pass
    
    @abstractmethod
    def prepare_model(self):
        pass
    
    @abstractmethod
    def train_step(self, data_dict, current_epoch):
        pass
    
    @abstractmethod
    def val_step(self, data_dict):
        pass
    
    def fit(self, resume_from_checkpoint=None):
        now = datetime.datetime.now()
        project_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(project_name)
        
        self.prepare_data()
        
        self.prepare_model()
        
        self.model, self.optimizer, self.lr_scheduler, ddp_train_loader, ddp_val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_loader, self.val_loader
        )
        
        global_step = 0
        for epoch in tqdm(range(self.num_train_epochs), desc="Epochs", unit="epoch", disable=not self.accelerator.is_main_process):
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            
            with tqdm(ddp_train_loader, desc=f"Epoch {epoch+1}/{self.num_train_epochs}", unit="step", disable=not self.accelerator.is_main_process) as pbar:
                for step, data_dict in enumerate(pbar):
                    with self.accelerator.accumulate(self.model):
                        loss_dict, grad_norm = self.train_step(data_dict, current_epoch=epoch)
                    
                    if self.accelerator.sync_gradients:
                        global_step += 1
                        logs = {**loss_dict, "lr": self.lr_scheduler.get_last_lr()[0], "grad_norm": grad_norm}
                        if global_step % self.log_steps == 0 and self.accelerator.is_main_process:
                            self.accelerator.log(logs, step=global_step)
                            pbar.set_postfix({
                                "loss": f"{loss_dict.get('loss', 0):.4f}",
                                "infonce_loss": f"{loss_dict.get('infonce_loss', 0):.4f}",
                                "grad_norm": f"{grad_norm:.4f}"
                            })
                    
                    epoch_loss += loss_dict.get('loss', 0)
                    epoch_steps += 1
            
            if self.accelerator.is_main_process:
                avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
                self.logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            if (epoch + 1) % self.val_epochs == 0 or epoch == self.num_train_epochs - 1:
                self.validate(ddp_val_loader, epoch)
            
            if (epoch + 1) % self.ckpt_epochs == 0 or epoch == self.num_train_epochs - 1:
                self.save_checkpoint(epoch)
        
        self.accelerator.end_training()
    
    def validate(self, val_loader, epoch):
        pass
    
    def save_checkpoint(self, epoch):
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.output_dir, "ckpt", f"epoch-{epoch}")
            self.accelerator.save_state(save_path)
            self.logger.info(f"Saved state to {save_path}")
