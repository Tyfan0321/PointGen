import hydra
from omegaconf import DictConfig
import torch
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler

from src.data.dataset_factory import DatasetFactory
from src.models.transformer_regtr_v0 import RegTrGenerative
from src.engine.processor_model import create_point_cloud_processor
from src.engine.evaluator import DiffusionEvaluator

@hydra.main(version_base=None, config_path="./config", config_name="config_v0")
def main(cfg: DictConfig):
    accelerator = Accelerator()
    print("Preparing evaluation data...")
    
    val_datasets = DatasetFactory.create(cfg.data, seqs="test")
    if not isinstance(val_datasets, list):
        val_datasets = [val_datasets]
        
    val_loaders = []
    for val_dataset in val_datasets:
        val_loaders.append(torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=1, 
            num_workers=cfg.num_workers, 
            shuffle=False, 
            pin_memory=True
        ))
    
    print("Preparing model...")
    encoder_config = cfg.encoder.encoder
    model = RegTrGenerative(
        encoder_config=encoder_config,
        **cfg.model,
        **cfg.loss
    )
    
    processor_type = encoder_config.get("type", "sonata")
    processor_config = cfg.encoder.processor
    processor = create_point_cloud_processor(
        processor_type,
        **processor_config
    )
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler(**cfg.scheduler)
    
    model = accelerator.prepare(model)
    val_loaders = [accelerator.prepare(loader) for loader in val_loaders]
    
    if cfg.resume:
        print(f"Loading checkpoint from {cfg.resume}...")
        accelerator.load_state(cfg.resume)
        print("Checkpoint loaded.")
    else:
        print("Warning: No checkpoint provided for evaluation. Using initialized weights.")
        
    model.eval()
    
    evaluator = DiffusionEvaluator(
        cfg, 
        processor=processor, 
        noise_scheduler=noise_scheduler
    )
    
    print("Starting evaluation...")
    if accelerator.is_main_process:
        with torch.no_grad():
            for i, val_loader in enumerate(val_loaders):
                seq_name = "3DMatch" if i == 0 else "3DLoMatch"
                print(f"Evaluating {seq_name}...")
                gen_log = evaluator.evaluate(model, val_loader)
                print(f"{seq_name} Evaluation Metrics: {gen_log}")
                
                import os
                import json
                from pathlib import Path
                if cfg.resume:
                    resume_path = Path(cfg.resume)
                    # Parse: .../3dmatch/2026-03-25/ckpt/epoch-124 -> .../3dmatch/eval
                    output_dir = resume_path.parent.parent.parent / "eval"
                    gen_log["ckpt"] = str(cfg.resume)
                    gen_log["num_inference_steps"] = cfg.num_inference_steps
                    gen_log["seqs"] = seq_name
                else:
                    output_dir = Path(getattr(cfg, "output_dir", "./outputs/eval_results"))
                
                os.makedirs(output_dir, exist_ok=True)
                result_path = os.path.join(output_dir, f"metrics_{seq_name}.jsonl")
                
                with open(result_path, 'a') as f:
                    f.write(json.dumps(gen_log) + '\n')
                
                print(f"Evaluation results saved to {result_path}")

if __name__ == "__main__":
    main()