from yacs.config import CfgNode
from pathlib import Path
from typing import Dict

from keepflow.models import ModelTemplate


class COPY_LAST(ModelTemplate):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        
        self.state = cfg.DATA.TRAJ.STATE
        self.pred_state = cfg.DATA.TRAJ.PRED_STATE
        
        assert self.state == 'state_pva'
        
        if self.pred_state == 'state_p':
            self.idx = [0, 1]
        elif self.pred_state == 'state_v':
            self.idx = [2, 3]
        else:
            raise ValueError
    
    def predict(self, data_dict, return_prob=False) -> Dict:
        size = data_dict["gt"].size()
        data_dict[("pred", 0)] = data_dict["obs"][:, -1:, self.idx].expand(size).contiguous()
        return data_dict
    
    def save(self, epoch: int = 0, path: Path = None) -> None:
        pass
    
    def load(self, path: Path = None) -> int:
        pass