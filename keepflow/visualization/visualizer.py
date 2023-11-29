from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path

from yacs.config import CfgNode
import numpy as np


class Visualizer(ABC):
    def __init__(self, cfg: CfgNode):
        self.output_dir = Path(cfg.SAVE_DIR) / "visualize"
        self.output_dir.mkdir(exist_ok=True)

    @abstractmethod
    def __call__(self, dict_list: List[Dict]) -> None:
        pass

    def to_numpy(self, tensor) -> np.ndarray:
        return tensor.cpu().numpy()  