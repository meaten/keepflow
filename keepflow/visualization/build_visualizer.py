from typing import Type
from yacs.config import CfgNode

from .visualizer import Visualizer
from .traj import TrajVisualizer
from .video import VideoVisualizer
from .motion import MotionVisualizer

def build_visualizer(cfg: CfgNode) -> Type[Visualizer]:
    if cfg.DATA.TASK == "traj":
        return TrajVisualizer(cfg)

    if cfg.DATA.TASK == "video":
        return VideoVisualizer(cfg)

    if cfg.DATA.TASK == "motion":
        return MotionVisualizer(cfg)
    