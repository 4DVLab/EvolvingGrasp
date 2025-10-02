from .DexGraspNet import DexGraspNet
from .Unidexgrasp import Unidexgrasp
from .multidex_shadowhand_ur import MultiDexShadowHandUR
from .DexGRAB import DexGRAB
# from .Grasp_anyting import Grasp_anyting
from typing import Any, Tuple, Dict
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from omegaconf import DictConfig, OmegaConf
import transforms3d
from datasets.misc import collate_fn_squeeze_pcd_batch_grasp
from datasets.transforms import make_default_transform
from datasets.base import DATASET
import json
from utils.registry import Registry

@DATASET.register()
class Combined_grasp(Dataset):
    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, case_only: bool=False, **kwargs: Dict) -> None:
        super(Combined_grasp, self).__init__()
        
        # Initialize each dataset with its specific configuration
        dataset1 = MultiDexShadowHandUR(cfg.dataset1_cfg, phase, slurm, case_only, **kwargs)
        dataset2 = DexGraspNet(cfg.dataset2_cfg, phase, slurm, case_only, **kwargs)
        dataset3 = Unidexgrasp(cfg.dataset3_cfg, phase, slurm, case_only, **kwargs)
        dataset4 = DexGRAB(cfg.dataset4_cfg, phase, slurm, case_only, **kwargs)
        # dataset5 = Grasp_anyting(cfg.dataset5_cfg, phase, slurm, case_only, **kwargs)
        # Combine the datasets
        self.combined_dataset = ConcatDataset([dataset1, dataset2, dataset3, dataset4]) #, dataset5
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, index: Any) -> Tuple:
        return self.combined_dataset[index]
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self.combined_dataset, **kwargs)

if __name__ == '__main__':
    config_path = "/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/configs/task/grasp_gen_ur.yaml"
    cfg = OmegaConf.load(config_path)
    

    
    dataloader = Combined_grasp(cfg, 'train', False).get_dataloader(batch_size=4,
                                                                             collate_fn=collate_fn_squeeze_pcd_batch_grasp,
                                                                             num_workers=0,
                                                                             pin_memory=True,
                                                                             shuffle=True,)

    device = 'cuda'
    for it, data in enumerate(dataloader):
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device)
        print()
