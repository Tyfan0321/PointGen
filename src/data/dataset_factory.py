import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from . import KittiDataset, IndoorDataset, IndoorTestDataset


def _get_config_value(config, key, default=None):
    try:
        return config.get(key, default)
    except AttributeError:
        return getattr(config, key, default)


class DatasetFactory:
    @staticmethod
    def create(data_config, seqs="train"):
        """
            seqs: "train" or "val" or "test"
        """
        dataset_type = data_config.dataset_type
        
        if dataset_type == "kitti":
            dataset = KittiDataset(
                seqs=seqs,
                root=data_config.root,
                data_list=data_config.data_list,
                npoints=data_config.npoints,
                voxel_size=data_config.voxel_size,
                augment=data_config.augment
            )
        elif dataset_type == "3dmatch":
            if seqs == "test":
                dataset = [
                    IndoorTestDataset(
                        seqs="3DMatch",
                        root=data_config.root,
                        data_list=data_config.data_list,
                        npoints=data_config.npoints,
                        voxel_size=data_config.voxel_size,
                    ), 
                    IndoorTestDataset(
                        seqs="3DLoMatch",
                        root=data_config.root,
                        data_list=data_config.data_list,
                        npoints=data_config.npoints,
                        voxel_size=data_config.voxel_size,
                    )
                ]
            else:
                dataset = IndoorDataset(
                    seqs=seqs,
                    root=data_config.root,
                    data_list=data_config.data_list,
                    npoints=data_config.npoints,
                    voxel_size=data_config.voxel_size,
                    augment=data_config.augment,
                    augmentation=_get_config_value(data_config, "augmentation", None),
                )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return dataset
