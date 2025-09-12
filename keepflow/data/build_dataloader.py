from typing import Tuple
from yacs.config import CfgNode
import numpy as np
from torch.utils.data import DataLoader

from .traj import build_traj_dataset
from .motion import build_motion_dataset
from .video import build_video_dataset


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg: CfgNode, rand=True, split="train", batch_size=None) -> DataLoader:
    if cfg.DATA.TASK == "traj":
        dataset, collate_fn = build_traj_dataset(cfg, split)
    elif cfg.DATA.TASK == "motion":
        dataset, collate_fn = build_motion_dataset(cfg, split)
    elif cfg.DATA.TASK == "video":
        dataset, collate_fn = build_video_dataset(cfg, split)
    else:
        raise ValueError        
    
    if batch_size is None:
        batch_size = cfg.SOLVER.BATCH_SIZE
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=rand,
        num_workers=min(cfg.SOLVER.NUM_WORKERS, len(dataset)),
        collate_fn=collate_fn,
        drop_last=True if split == 'train' else False,
        worker_init_fn=worker_init_fn,
        )
    
    return loader
        