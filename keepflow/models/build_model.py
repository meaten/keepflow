from abc import abstractmethod
from typing import Dict
from copy import deepcopy
from pathlib import Path
from yacs.config import CfgNode
import torch
import torch.nn as nn

from keepflow.models.traj import build_traj_pred_model
from keepflow.models.video import build_video_pred_model
from keepflow.models.motion import build_motion_pred_model


def build_model(cfg: CfgNode) -> nn.Module:
    if cfg.MODEL.TYPE == "GT":
        return GT(cfg)
    elif cfg.MODEL.TYPE == "COPY_LAST":
        return COPY_LAST(cfg)

    if cfg.DATA.TASK == "traj":
        return build_traj_pred_model(cfg)
    elif cfg.DATA.TASK == "video":
        return build_video_pred_model(cfg)
    elif cfg.DATA.TASK == "motion":
        return build_motion_pred_model(cfg)
    else:
        raise(ValueError)


class ModelTemplate(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(ModelTemplate, self).__init__()
        
        self.device = cfg.DEVICE

        self.save_dir = Path(cfg.SAVE_DIR)
        self.model_path = self.save_dir / "ckpt.pt"

        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len = cfg.DATA.PREDICT_LENGTH
        
    @abstractmethod
    def predict_from_new_obs(self, data_dict: Dict, time_step: int) -> Dict:
        return data_dict

    @abstractmethod
    def predict(self, data_dict: Dict, return_prob=False) -> Dict:
        pass

    @abstractmethod
    def update(self, data_dict) -> None:
        pass
    
    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.model_path
            
        ckpt = {
            'epoch': epoch,
            'state': self.state_dict(),
            'optim_state': self.optimizer.state_dict(),
        }

        torch.save(ckpt, path)
    

    def load(self, path: Path=None) -> int:
        if path is None:
            path = self.model_path
        
        ckpt = torch.load(path)
        self.load_state_dict(ckpt['state'])

        self.optimizer.load_state_dict(ckpt['optim_state'])
    
        return ckpt['epoch']
    
    def check_saved_path(self, path: Path = None) -> bool:
        if path is None:
            path = self.model_path

        return path.exists()
    

class GT(ModelTemplate):
    def predict(self, data_dict, return_prob=False) -> Dict:
        data_dict[("pred", 0)] = deepcopy(data_dict["gt"])
        return data_dict
    
    def save(self, epoch: int = 0, path: Path = None) -> None:
        pass
    
    def load(self, path: Path = None) -> int:
        pass
    

class COPY_LAST(ModelTemplate):
    def predict(self, data_dict, return_prob=False) -> Dict:
        size = data_dict["gt"].size()
        data_dict[("pred", 0)] = data_dict["obs"][:, -1:].expand(size).contiguous()
        return data_dict
    
    def save(self, epoch: int = 0, path: Path = None) -> None:
        pass
    
    def load(self, path: Path = None) -> int:
        pass