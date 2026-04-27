import hydra
from omegaconf import DictConfig
from src.engine import DiffusionTrainer


@hydra.main(version_base=None, config_path="./config", config_name="config_v0")
def main(cfg: DictConfig):
    print("Creating DiffusionTrainer...")
    trainer = DiffusionTrainer(cfg)
    
    print("Starting training...")
    trainer.fit(resume_from_checkpoint=cfg.resume)

if __name__ == "__main__":
    main()
