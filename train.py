import hydra
from omegaconf import DictConfig
from src.engine import DiffusionTrainer


@hydra.main(version_base=None, config_path="./config", config_name="config_bak")
def main(cfg: DictConfig):
    print("Creating DiffusionTrainer...")
    trainer = DiffusionTrainer(cfg)
    
    if cfg.eval_only:
        trainer.prepare_data()
        trainer.prepare_model()
        trainer.model, trainer.optimizer, trainer.lr_scheduler, ddp_train_loader, ddp_val_loader = trainer.accelerator.prepare(
            trainer.model, trainer.optimizer, trainer.lr_scheduler, trainer.train_loader, trainer.val_loader
        )
        print("Running evaluation only...")
        if cfg.resume:
            trainer.accelerator.load_state(cfg.resume)
            print(f"Loaded checkpoint from {cfg.resume}")
        trainer.validate(ddp_val_loader, epoch=0)
    else:
        print("Starting training...")
        trainer.fit(resume_from_checkpoint=cfg.resume)


if __name__ == "__main__":
    main()
