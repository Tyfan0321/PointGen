import os
import yaml
import json
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置对象
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def save_config_snapshot(config: DictConfig, output_dir: str):
    """
    保存配置快照，确保实验可重现性
    
    Args:
        config: 配置对象
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    yaml_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(yaml_path, 'w') as f:
        OmegaConf.save(config, f)
    
    print(f"Configuration snapshot saved to {yaml_path}")
