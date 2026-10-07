import sys
from typing import Dict, List
from yacs.config import CfgNode
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim

from keepflow.models import ModelTemplate
from keepflow.data import build_dataloader
sys.path.append('extern/traj/SingularTrajectory')
import baseline
from SingularTrajectory import *
from utils import *


# AA = Adaptive Anchor


class SingularTrajectory_wo_AA(ModelTemplate):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.num_samples = cfg.TEST.N_TRIAL
        hyper_params = get_exp_config("extern/traj/SingularTrajectory/config/config_example.json")
        self.hyper_params = hyper_params
        model_cfg = DotDict({'scheduler': 'ddim', 'steps': 10, 'beta_start': 1.e-4, 'beta_end': 5.e-2, 'beta_schedule': 'linear', 
                       'k': hyper_params.k, 's': hyper_params.num_samples})
        self.baseline = getattr(baseline, hyper_params.baseline).TrajectoryPredictor(model_cfg)
        hook_func = DotDict({"model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
                            "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
                            "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook})
        self.model = SingularTrajectory(self.baseline, hook_func, hyper_params)
        
        obs_traj_all = []
        gt_traj_all = []
        loader = build_dataloader(cfg)  # train split
        print("Loading data for Singular Space...")
        for batch in tqdm(loader):
            obs_traj_all.append(batch['obs'].clone())
            gt_traj_all.append(batch['gt'].clone())
        obs_traj_all = torch.cat(obs_traj_all, dim=0)
        gt_traj_all = torch.cat(gt_traj_all, dim=0)
        try:
            self.model.calculate_parameters(obs_traj_all, gt_traj_all)
        except ValueError:
            # for synthesized datasets
            g = torch.Generator(device=obs_traj_all.device)
            g.manual_seed(123)
            noise = torch.rand(
                obs_traj_all.shape, 
                generator=g, 
                device=obs_traj_all.device, 
                dtype=obs_traj_all.dtype
            )
            obs_traj_all += 2.0 * noise
            self.model.calculate_parameters(obs_traj_all, gt_traj_all)
            
        self.optimizer = optim.Adam(self.parameters(), lr=hyper_params.lr, weight_decay=hyper_params.weight_decay)
        self.optimizers = [self.optimizer]

    def create_batch(self, data_dict: Dict) -> List:
        obs = data_dict['obs']
        neighbors = data_dict['neighbors']
        
        if len(neighbors) != 0:
            past_traj = []
            for o, n in zip(obs, neighbors):
                past_traj.append(torch.cat([o[None], n[~torch.isnan(n[..., 0]).any(dim=1)]], dim=0))
            partitioning_index = np.cumsum([0] + [len(n) for n in past_traj[:-1]])
            obs_traj = torch.cat(past_traj, dim=0)
            batch_size = len(obs_traj)
            traj_mask = torch.zeros(batch_size, batch_size).to(self.device)
            for i in range(len(partitioning_index) - 1):
                traj_mask[partitioning_index[i]:partitioning_index[i+1], partitioning_index[i]:partitioning_index[i+1]] = 1.
            traj_mask[partitioning_index[-1]:batch_size, partitioning_index[-1]:batch_size] = 1
            
        else:  # no neighbors
            batch_size = data_dict['obs'].shape[0]
            partitioning_index = np.arange(batch_size)
            traj_mask = torch.eye(batch_size).to(self.device)
            obs_traj = obs
            
        pred_traj = data_dict['gt']
        adaptive_anchor = torch.zeros((obs_traj.size(0), self.model.k, self.model.s), dtype=torch.float, device=obs_traj.device)
        addl_info = DotDict({"scene_mask": traj_mask,
                             "num_samples": self.num_samples,
                             "anchor": adaptive_anchor})
        return obs_traj, adaptive_anchor, None, addl_info, partitioning_index
        
    def training_step(self, data_dict) -> None:
        obs_traj, adaptive_anchor, pred_traj, addl_info, index = self.create_batch(data_dict)
        output = self.model(obs_traj, adaptive_anchor, pred_traj, addl_info)
    
        loss = (output["recon_traj"][:, index] - data_dict['gt'].unsqueeze(dim=0)).norm(p=2, dim=-1)
        loss = loss.mean(dim=-1).min(dim=0)[0].mean()
        loss[torch.isnan(loss)] = 0

        self.optimizer.zero_grad()
        loss.backward()
        if self.hyper_params.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def predict(self, data_dict: Dict, return_prob=False) -> Dict:        
        obs_traj, adaptive_anchor, pred_traj, addl_info, index = self.create_batch(data_dict)
        
        # cannot change num_samples. deterministic sampling
        output = self.model(obs_traj, adaptive_anchor + torch.rand_like(adaptive_anchor), pred_traj, addl_info)
        
        data_dict[('pred', 0)] = output['recon_traj'][:, index].transpose(0, 1)
        
        if return_prob:
            trajs = [data_dict[('pred', 0)]]
            # num_samples trajectories per call; repeat until the sample budget is met.
            n_repeat = max(0, -(-self.kde_n_sample // self.num_samples) - 1)
            for _ in range(n_repeat):
                output = self.model(obs_traj, adaptive_anchor + torch.rand_like(adaptive_anchor), pred_traj, addl_info)
                traj_pred = output['recon_traj'][:, index].transpose(0, 1)
                trajs.append(traj_pred)
            trajs = torch.cat(trajs, dim=1)
            trajs = torch.cat([trajs, torch.zeros_like(trajs)], dim=3)
            data_dict[("prob", 0)] = trajs[..., :3]

        return data_dict