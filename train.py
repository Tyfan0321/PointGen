import hydra
from omegaconf import DictConfig
from src.engine import DiffusionTrainer


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    # 添加 resume 和 eval_only 参数
    import argparse
    parser = argparse.ArgumentParser(description="PointGen 训练脚本")
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None, 
        help="从检查点恢复训练，指定检查点路径或 'latest'"
    )
    parser.add_argument(
        "--eval_only", 
        action="store_true", 
        default=False, 
        help="仅运行评估，不进行训练"
    )
    args = parser.parse_args()
    print("Creating DiffusionTrainer...")
    trainer = DiffusionTrainer(cfg)
    
    print("Preparing data and model...")
    trainer.prepare_data()
    trainer.prepare_model()
    
    trainer.model, trainer.optimizer, trainer.lr_scheduler, ddp_train_loader, ddp_val_loader = trainer.accelerator.prepare(
        trainer.model, trainer.optimizer, trainer.lr_scheduler, trainer.train_loader, trainer.val_loader
    )
    
    if args.eval_only:
        print("Running evaluation only...")
        if args.resume:
            trainer.accelerator.load_state(args.resume)
            print(f"Loaded checkpoint from {args.resume}")
        trainer.validate(ddp_val_loader, epoch=0)
    else:
        print("Starting training...")
        trainer.fit(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
