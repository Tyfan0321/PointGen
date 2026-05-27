import hydra
from omegaconf import DictConfig
import torch
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler
import json
import os
from datetime import datetime
from pathlib import Path

from src.data.dataset_factory import DatasetFactory
from src.models.transformer_regtr_v0 import RegTrGenerative
from src.engine.processor_model import create_point_cloud_processor
from src.engine.evaluator import DiffusionEvaluator


def _safe_path_part(value):
    return str(value).replace("/", "-").replace(" ", "_")


def _get_eval_output_dir(cfg):
    if cfg.resume:
        resume_path = Path(cfg.resume)
        run_dir = resume_path.parent.parent
        ckpt_name = resume_path.name
    else:
        run_dir = Path(getattr(cfg, "output_dir", "./outputs/eval_results"))
        ckpt_name = "init"

    sample_part = "full" if cfg.num_gen_samples is None else f"samples-{cfg.num_gen_samples}"
    config_name = "_".join([
        f"steps-{cfg.num_inference_steps}",
        sample_part,
        f"scheduler-{_safe_path_part(cfg.inference_type)}",
        f"overlap-{'on' if cfg.eval.get('use_overlap_metrics', False) else 'off'}",
        f"seed-{cfg.seed}",
    ])
    return run_dir / "eval" / ckpt_name / config_name


def _build_eval_record(cfg, seq_name, gen_log, val_loader):
    dataset_size = len(val_loader.dataset) if hasattr(val_loader, "dataset") else None
    evaluated_samples = dataset_size
    if cfg.num_gen_samples is not None and dataset_size is not None:
        evaluated_samples = min(cfg.num_gen_samples, dataset_size)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": cfg.experiment_name,
        "runname": cfg.runname,
        "ckpt": str(cfg.resume) if cfg.resume else None,
        "seqs": seq_name,
        "dataset_size": dataset_size,
        "evaluated_samples": evaluated_samples,
        "num_gen_samples": cfg.num_gen_samples,
        "num_inference_steps": cfg.num_inference_steps,
        "inference_type": cfg.inference_type,
        "use_overlap_metrics": cfg.eval.get("use_overlap_metrics", False),
        "seed": cfg.seed,
        "metrics": gen_log,
    }

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
    output_dir = _get_eval_output_dir(cfg)
    
    print("Starting evaluation...")
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Evaluation results will be saved to {output_dir}")
        with torch.no_grad():
            for i, val_loader in enumerate(val_loaders):
                seq_name = "3DMatch" if i == 0 else "3DLoMatch"
                print(f"Evaluating {seq_name}...")
                gen_log = evaluator.evaluate(model, val_loader)
                print(f"{seq_name} Evaluation Metrics: {gen_log}")

                record = _build_eval_record(cfg, seq_name, gen_log, val_loader)
                summary_path = output_dir / f"{seq_name}_summary.json"
                metrics_path = output_dir / "metrics.jsonl"

                with open(summary_path, "w") as f:
                    json.dump(record, f, indent=2)

                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                
                print(f"{seq_name} summary saved to {summary_path}")
        print(f"All evaluation metrics appended to {output_dir / 'metrics.jsonl'}")

if __name__ == "__main__":
    main()
