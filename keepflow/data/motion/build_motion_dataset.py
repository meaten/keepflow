from typing import Tuple
from collections.abc import Callable
from yacs.config import CfgNode
from torch.utils.data import Dataset


def build_motion_dataset(cfg: CfgNode, split: str) -> Tuple[Dataset, Callable]:
    # train, val, test
    if cfg.DATA.DATASET_NAME == "h36motion":
        import os
        from .h36motion import H36motion
        dataset_train = H36motion(
            path_to_data=os.path.join(cfg.DATA.PATH, cfg.DATA.TASK, "h3.6m", "dataset"),
            actions="all",
            input_n=cfg.DATA.OBSERVE_LENGTH,
            output_n=cfg.DATA.PREDICT_LENGTH,
            split=0,
            load_3d=False)

        if split == "train":
            dataset = dataset_train
        elif split == "val":
            dataset_val = H36motion(
                path_to_data=os.path.join(cfg.DATA.PATH, cfg.DATA.TASK, "h3.6m", "dataset"),
                actions="smoking",
                input_n=cfg.DATA.OBSERVE_LENGTH,
                output_n=cfg.DATA.PREDICT_LENGTH,
                split=2,
                data_mean=dataset_train.data_mean,
                data_std=dataset_train.data_std,
                onehotencoder=dataset_train.onehotencoder,
                load_3d=False)

            dataset = dataset_val
        elif split == "test":
            dataset_test = H36motion(
                path_to_data=os.path.join(cfg.DATA.PATH, cfg.DATA.TASK, "h3.6m", "dataset"),
                actions="smoking",
                input_n=cfg.DATA.OBSERVE_LENGTH,
                output_n=cfg.DATA.PREDICT_LENGTH,
                split=1,
                data_mean=dataset_train.data_mean,
                data_std=dataset_train.data_std,
                onehotencoder=dataset_train.onehotencoder,
                load_3d=False)

            dataset = dataset_test
        
    from .h36motion import seq_collate
        
    return dataset, seq_collate