from yacs.config import CfgNode
from typing import Callable


def build_metrics(cfg: CfgNode) -> Callable:
    if cfg.DATA.TASK == "traj":
        from .traj.traj_metrics import TrajMetrics
        return TrajMetrics(cfg)

    if cfg.DATA.TASK == "video":
        from .video.video_metrics import video_metrics
        return video_metrics

    if cfg.DATA.TASK == "motion":
        from .motion.motion_metrics import motion_metrics
        return motion_metrics