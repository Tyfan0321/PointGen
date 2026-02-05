import math
import torch
import torch.optim as optim
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler

from src.engine.base_trainer import BaseTrainer
from src.engine.data_processor import DiffusionDataProcessor
from src.engine.evaluator import DiffusionEvaluator
from src.data.dataset_factory import DatasetFactory
from src.models.transformer_regtr import RegTrGenerative
from src.engine.model_processor import PointCloudProcessor


class DiffusionTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.data_processor = DiffusionDataProcessor(cfg)
        self.evaluator = DiffusionEvaluator(cfg)
        self.feat_stop_epoch = cfg.feat_stop_epoch
    
    def prepare_data(self):
        train_dataset = DatasetFactory.create(self.cfg.data, seqs="train")
        val_dataset = DatasetFactory.create(self.cfg.data, seqs="val")
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.train_batch_size, 
            num_workers=self.cfg.num_workers, 
            shuffle=True, 
            pin_memory=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.train_batch_size, 
            num_workers=self.cfg.num_workers, 
            shuffle=False, 
            pin_memory=True
        )
    
    def prepare_model(self):
        self.model = RegTrGenerative(**self.cfg.model, **self.cfg.loss)
        self.processor = PointCloudProcessor(**self.cfg.processor)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg.optim.lr, 
            betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2), 
            eps=self.cfg.optim.adam_eps,
            weight_decay=self.cfg.optim.adam_weight_decay
        )
        
        num_training_steps = self.num_train_epochs * math.ceil(len(self.train_loader) / self.gradient_accumulation_steps)
        self.lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler.name,
            self.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=self.cfg.lr_scheduler.lr_warmup_steps * self.accelerator.num_processes,
            num_cycles=self.cfg.lr_scheduler.lr_num_cycles,
        )
        
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(**self.cfg.scheduler)
    
    def train_step(self, data_dict, current_epoch):
        train_data_dict = self.data_processor.prepare_noisy_data(data_dict, self.noise_scheduler)
        
        model_output = self.model(**train_data_dict)
        
        loss_dict = self.data_processor.compute_loss(model_output, train_data_dict, current_epoch, self.feat_stop_epoch)
        
        self.accelerator.backward(loss_dict["overall_loss"])
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        return loss_dict
    
    def val_step(self, data_dict):
        val_data_dict = self.data_processor.prepare_noisy_data(data_dict, self.noise_scheduler)
        
        with torch.no_grad():
            model_output = self.model(**val_data_dict)
        
        loss_dict = self.data_processor.compute_loss(model_output, val_data_dict, 0, 0)
        
        return loss_dict
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        total_infonce_loss = torch.tensor(0.0, device=self.accelerator.device)
        num_samples = torch.tensor(0, device=self.accelerator.device)
        
        for step, data_dict in enumerate(val_loader):
            val_loss_dict = self.val_step(data_dict)
            total_loss += val_loss_dict["loss"]
            total_infonce_loss += val_loss_dict["infonce_loss"]
            num_samples += 1
        
        gathered_losses = self.accelerator.gather(total_loss)
        gathered_infonce_loss = self.accelerator.gather(total_infonce_loss)
        gathered_samples = self.accelerator.gather(num_samples)
        
        if self.accelerator.is_main_process:
            global_loss = gathered_losses.sum() / gathered_samples.sum()
            global_infonce_loss = gathered_infonce_loss.sum() / gathered_samples.sum()
            val_log = {
                "val_loss": global_loss.item(),
                "val_infonce_loss": global_infonce_loss.item()
            }
            self.logger.info(f"Epoch {epoch + 1}, Validation Loss: {global_loss.item():.4f}")
            self.accelerator.log(val_log, step=epoch + 1)
            
            if self.do_gen:
                gen_log = self.evaluator.evaluate(self.model, self.processor, self.noise_scheduler, val_loader)
                self.logger.info(f"Epoch {epoch + 1}, Generation Metrics: {gen_log}")
                self.accelerator.log(gen_log, step=epoch + 1)
