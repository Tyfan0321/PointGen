import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import json
from typing import Dict
from munch import munchify

from data.threedmatch_data import IndoorDataset, IndoorTestDataset
from models.models.regtr import REGTR, Evaluator
from engine.trainer import EpochBasedTrainer


class OverallLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.weight_feat_loss = cfg.weight_feat_loss
        self.weight_conf_loss = cfg.weight_conf_loss
    
    def forward(self, l_feat, l_corr, l_conf) -> Dict[str, torch.Tensor]:
        loss = l_feat * self.weight_feat_loss + l_conf * self.weight_conf_loss + l_corr
        return {'loss':loss, 'l_feat':l_feat, 'l_corr':l_corr, 'l_conf':l_conf}


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        train_dataset = IndoorDataset(
            cfg.data.root, 'train', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, cfg.data.augment)
        val_dataset1 = IndoorTestDataset(cfg.data.root, "3DMatch", cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, True)
        val_dataset2 = IndoorTestDataset(cfg.data.root, "3DLoMatch", cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, True)

        self.train_loader = DataLoader(train_dataset, 1, num_workers=cfg.data.num_workers, shuffle=True, pin_memory=True)
        self.val_loader = [
            DataLoader(val_dataset1, 1, num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True),
            DataLoader(val_dataset2, 1, num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True),
        ]
        self.model = REGTR(cfg.model).cuda()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.optim.step_size, gamma=cfg.optim.gamma)
        self.loss_func = OverallLoss(cfg.loss).cuda()
        self.evaluator = Evaluator(cfg.eval).cuda()
    
    def step(self, data_dict) -> Dict[str,torch.Tensor]:
        output_dict = self.model(*data_dict[:4])
        output_dict['gt_transform'] = data_dict[4][0]
        if len(data_dict) > 5:
            output_dict['covariance'] = data_dict[5][0]
        loss = self.model.compute_loss(output_dict)
        loss_dict: Dict = self.loss_func(*loss)
        result_dict = self.evaluator(output_dict)
        loss_dict.update(result_dict)
        return loss_dict



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", choices=["train", "test"])
    parser.add_argument("--config", default='./config/3dmatch_regtr1.json', type=str)
    parser.add_argument("--resume_epoch", default=0, type=int)
    parser.add_argument("--resume_log", default=None, type=str)
    parser.add_argument("--load_pretrained", default='regtr1-epoch-30', type=str)

    _args = parser.parse_args()

    with open(_args.config, 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
    
    if _args.mode == "train":
        Trainer(args).fit(_args.resume_epoch, _args.resume_log)
    elif _args.mode == "test":
        tester = Trainer(args)
        tester.load_snapshot(_args.load_pretrained)
        # e.g. tester.load_snapshot("cast-epoch-40")
        if isinstance(tester.val_loader, list):
            for loader in tester.val_loader:
                tester.validate_epoch(loader)
        else: tester.validate_epoch(tester.val_loader)
        tester.validate_epoch()
    else: assert "Unspecified mode."
