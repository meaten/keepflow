from typing import Tuple
from collections.abc import Callable
from pathlib import Path
from yacs.config import CfgNode
from torch.utils.data import Dataset


def build_video_dataset(cfg: CfgNode, split: str) -> Tuple[Dataset, Callable]:
    # only train, test
    from .datasets_factory import video_dataset
    if cfg.DATA.DATASET_NAME == "bair":
        path = Path(cfg.DATA.PATH) / cfg.DATA.TASK
        img_width = 64
    elif cfg.DATA.DATASET_NAME == "kth":
        path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "kth_action"
        img_width = 128
    elif cfg.DATA.DATASET_NAME == "mnist":
        path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "moving-mnist-example" / f"moving-mnist-{split}.npz"
        img_width = 64
    dataset = video_dataset(dataset_name=cfg.DATA.DATASET_NAME,
                        data_path=path,
                        split=split,
                        img_width=img_width,
                        input_n=cfg.DATA.OBSERVE_LENGTH,
                        output_n=cfg.DATA.PREDICT_LENGTH,
                        injection_action="concat")
    
    from .mnist import seq_collate
    
    return dataset, seq_collate