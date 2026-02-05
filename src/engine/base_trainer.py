import os
import datetime
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from src.utils.config_utils import save_config_snapshot


class BaseTrainer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.logging_dir = os.path.join(self.output_dir, cfg.logging_dir)
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        save_config_snapshot(cfg, self.output_dir)
        
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
        project_name = f"{self.cfg.experiment_name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(project_name)
        
        self.prepare_data()
        
        self.prepare_model()
        
        self.model, self.optimizer, self.lr_scheduler, ddp_train_loader, ddp_val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_loader, self.val_loader
        )
        
        global_step = 0
        for epoch in range(self.num_train_epochs):
            self.model.train()
            for step, data_dict in enumerate(ddp_train_loader):
                with self.accelerator.accumulate(self.model):
                    loss_dict = self.train_step(data_dict, current_epoch=epoch)
                
                if self.accelerator.sync_gradients:
                    global_step += 1
                    logs = {**loss_dict, "lr": self.lr_scheduler.get_last_lr()[0]}
                    if global_step % self.log_steps == 0 and self.accelerator.is_main_process:
                        self.accelerator.log(logs, step=global_step)
            
            if (epoch + 1) % self.val_epochs == 0 or epoch == self.num_train_epochs - 1:
                self.validate(ddp_val_loader, epoch)
            
            if (epoch + 1) % self.ckpt_epochs == 0 or epoch == self.num_train_epochs - 1:
                self.save_checkpoint(epoch)
        
        self.accelerator.end_training()
    
    def validate(self, val_loader, epoch):
        pass
    
    def save_checkpoint(self, epoch):
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
            self.accelerator.save_state(save_path)
            self.logger.info(f"Saved state to {save_path}")
