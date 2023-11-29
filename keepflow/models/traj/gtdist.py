import torch
from copy import deepcopy
from yacs.config import CfgNode
from typing import Dict
from pathlib import Path

from keepflow.models import ModelTemplate


class GT_Dist(ModelTemplate):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)        
        import dill
        env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "processed_data" / f"{cfg.DATA.DATASET_NAME}_train.pkl"
        with open(env_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')
            
        from .components.gmm2d import GMM2D
        gt_dist = train_env.gt_dist
        assert gt_dist is not None, "environment does not have GT distributions"
        gt_dist = gt_dist.transpose(1, 0, 2)
        gt_dist = gt_dist[-self.pred_len:]
        L, N, D = gt_dist.shape
        log_pis = torch.log(torch.ones(1, 1, L, N) / N)
        mus = torch.Tensor(gt_dist[None, None, ..., :2])
        log_sigmas = torch.log(torch.Tensor(gt_dist[None, None, ..., 2:]))
        corrs = torch.zeros(1, 1, L, N)
        
        self.kernels = GMM2D(log_pis.to(self.device),
                             mus.to(self.device),
                             log_sigmas.to(self.device),
                             corrs.to(self.device))
    
        self.obss = []
        self.gts = []
        
    def predict(self, data_dict: Dict, return_prob=False):
        data_dict[("pred", 0)] = deepcopy(data_dict["gt"])
        self.obss.append(data_dict["obs"].cpu().numpy())
        self.gts.append(data_dict["gt"].cpu().numpy())
        if return_prob:
            data_dict[("prob", 0)] = self.kernels
        return data_dict