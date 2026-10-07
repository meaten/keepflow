from pathlib import Path
from yacs.config import CfgNode
from typing import Dict

import numpy as np
import torch
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.optimal_transport import OTPlanSampler

from .continuousflowchain_onestep import (
    ContinuousFlowChainTraj,
    FlowChain_CNF,
    FlowChain_CNF_CIF,
)

# from .led import social_transformer


class FlowMatchingChainTraj(ContinuousFlowChainTraj):
    def _build_model_dict(self, cfg: CfgNode) -> Dict:
        model_dict = {
            "FlowMatchingChainOnestep": FlowMatchingChain_CNF,
            "FlowMatchingChainOnestep_spline": FlowMatchingChain_CNF_spline,
            "FlowMatchingChainOnestep_CIF": FlowMatchingChain_CNF_CIF,
            # "FlowMatchingChainOnestep_CIF_cross": FlowMatchingChain_CNF_CIF_cross,
            "FlowMatchingChainOnestep_CIF_spline": FlowMatchingChain_CNF_CIF_spline,
        }
            
        return model_dict
    
    def training_step(self, data_dict: Dict) -> Dict:
        node = self.get_node_type(data_dict)
        dist_args = self._encode(data_dict, node)
        
        gt = data_dict['gt']
        if self.dequantize:
            gt += torch.rand_like(data_dict['gt']) / 100

        base_pos = self._get_base_pos(data_dict)
        
        loss = self.flow[node].flow_matching(base_pos, gt.nan_to_num(), dist_args)
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.mean().item()}
    
    
# class FlowMatchingChainTraj_transformer(FlowMatchingChainTraj):
#     def _build_encoder(self, cfg: CfgNode) -> None:
#         self.encoder = social_transformer()
#         self.cond_shaper = torch.nn.Linear(256, cfg.MODEL.FLOW.COND_SIZE)
#         from keepflow.data.traj import EmptyEnv
#         env = EmptyEnv(cfg)
#         self.env = env
    
#     def _build_optimizer(self, cfg: CfgNode) -> None:
#         self.dequantize = cfg.SOLVER.DEQUANTIZE
        
#         decay_parameters = [
#             n for n, p in self.flow.named_parameters() if 'bias' not in n]
        
#         optimizer_grouped_parameters = [
#             {'params': self.encoder.parameters(), },
#             {"params": [p for n, p in self.flow.named_parameters() if n in decay_parameters], "weight_decay": cfg.SOLVER.WEIGHT_DECAY,},
#             {"params": [p for n, p in self.flow.named_parameters() if n not in decay_parameters], "weight_decay": 0.0,},
#             {"params": [p for n, p in self.cond_shaper.named_parameters() if n not in decay_parameters], "weight_decay": 0.0,},
#         ]
        
#         self.optimizer = torch.optim.AdamW(
#             optimizer_grouped_parameters, cfg.SOLVER.LR)
#         self.optimizers = [self.optimizer]
        
#     def _data_preprocess(self, data_dict):
#         obs_st = data_dict['obs']
#         neighbors_st = data_dict['neighbors']
        
#         if len(neighbors_st) != 0:
#             past_traj = []
#             for o, n in zip(obs_st, neighbors_st):
#                 past_traj.append(torch.cat([o[None], n[~torch.isnan(n[..., 0]).any(dim=1)]], dim=0))
#             partitioning_index = np.cumsum([0] + [len(n) for n in past_traj[:-1]])
#             past_traj = torch.cat(past_traj, dim=0)
#             batch_size = len(past_traj)
#             traj_mask = torch.zeros(batch_size, batch_size).to(self.device)
#             for i in range(len(partitioning_index) - 1):
#                 traj_mask[partitioning_index[i]:partitioning_index[i+1], partitioning_index[i]:partitioning_index[i+1]] = 1.
#             traj_mask[partitioning_index[-1]:batch_size, partitioning_index[-1]:batch_size] = 1
            
#         else:  # no neighbors
#             batch_size = data_dict['obs'].shape[0]
#             partitioning_index = np.arange(batch_size)
#             traj_mask = torch.eye(batch_size).to(self.device)
#             past_traj = obs_st
            
        
#         # past_traj = torch.nn.functional.pad(past_traj, (0, 0, 10 - self.obs_len, 0), 'replicate')
#         fut_traj = data_dict['gt']
#         return partitioning_index, batch_size, traj_mask, past_traj, fut_traj
        
#     def _encode(self, data_dict: Dict, node: str) -> torch.Tensor:
#         partitioning_index, batch_size, mask, past_traj, fut_traj = self._data_preprocess(data_dict)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         context = self.encoder(past_traj, mask)[partitioning_index]
#         context = self.cond_shaper(context.squeeze())
#         return context
    
#     def save(self, epoch: int = 0, path: Path=None) -> None:
#         if path is None:
#             path = self.model_path

#         ckpt = {
#             'epoch': epoch,
#             'encoder': self.encoder.state_dict(),
#             'cond_shaper': self.cond_shaper.state_dict(),
#             'flow': self.flow.state_dict(),
#             'optim_state': self.optimizer.state_dict()
#         }

#         torch.save(ckpt, path)

#     def load(self, path: Path=None) -> int:
#         if path is None:
#             path = self.model_path
        
#         ckpt = torch.load(path, map_location=self.device)
#         self.encoder.load_state_dict(ckpt['encoder'])
#         self.cond_shaper.load_state_dict(ckpt['cond_shaper'])
#         self.flow.load_state_dict(ckpt['flow'])
        
#         self.optimizer.load_state_dict(ckpt['optim_state'])
#         epoch = ckpt["epoch"]
                    
#         return epoch

    

class FlowMatchingChain_CNF(FlowChain_CNF):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_size: int,
                 **kwargs):
        super().__init__(input_size, n_blocks, hidden_size, n_hidden, cond_size, **kwargs)
        self.hybrid_loss = False
        self.FM = ConditionalFlowMatcher(sigma=self.base_sigma)
        # self.ot_sampler = OTPlanSampler(method='exact')
        
    def flow_matching(self, base_pos, seq, cond):
        sample = self.base_dist(base_pos, step=0).sample()
        x0 = torch.cat([sample[:, None], seq[:, :-1]], dim=1)
        x1 = seq

        batch_size, *_ = seq.shape
        
        loss = []
        self.net.vector_field.net.data_control.cond = cond
        for i in range(self.pred_len):
            t = sample_t(num_samples=batch_size, a=4, device=x0.device)
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0[:, i], x1[:, i], t=t)
            t += i
            vt = self.net.vector_field.net(t, xt)
            loss.append(((vt - ut) ** 2).mean(dim=-1))
        
        loss = torch.stack(loss, dim=1)
        
        if self.hybrid_loss:
            loss -= self.log_prob(base_pos, seq, cond)
            
        return loss
    
    
class FlowMatchingChain_CNF_CIF(FlowChain_CNF_CIF):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_size: int,
                 **kwargs):
        super().__init__(input_size, n_blocks, hidden_size, n_hidden, cond_size, **kwargs)
        self.hybrid_loss = False
        self.FM = ConditionalFlowMatcher(sigma=self.base_sigma)
        # self.ot_sampler = OTPlanSampler(method='exact')
        
    def flow_matching(self, base_pos, seq, cond):
        sample = self.base_dist(base_pos, step=0).sample()
        x0 = torch.cat([sample[:, None], seq[:, :-1]], dim=1)
        x1 = seq
        
        batch_size, *_ = seq.shape

        loss = []
        sample = self.base_dist(x0[:, 0], step=0).sample()
        # w ~ q(w|z_0) and the correction term +log q(w|z_0) - log r(w|z_1) follow
        # Eq. (19) of the manuscript, matching FlowMatchingChain_CNF_CIF_spline.
        # This class is the "w/o consistent rectified flow" ablation, so it must
        # differ from the spline variant *only* in the target vector field.
        u, log_q_u = self.cif.q_u_given_z.sample(torch.cat([sample, cond], dim=-1))
        for i in range(self.pred_len):
            t = sample_t(num_samples=batch_size, a=4, device=x0.device)
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0[:, i], x1[:, i], t=t)
            t += i
            self.net.vector_field.net.data_control.cond = u
            vt = self.net.vector_field.net(t, xt)
            loss.append((vt - ut).pow(2).mean(dim=-1))
        loss = torch.stack(loss, dim=1)
        log_r_u = self.cif.r_u_given_x.log_prob(u, torch.cat([x1[:, -1], cond], dim=-1))
        loss -= (-log_q_u + log_r_u)[:, None] / self.pred_len * 0.01

        if self.hybrid_loss:
            loss -= self.log_prob(base_pos, seq, cond)
            
        return loss
        
    
# class FlowMatchingChain_CNF_CIF_cross(FlowMatchingChain_CNF_CIF):    
#     def flow_matching(self, base_pos, seq, cond):
        
#         batch_size, *_ = seq.shape
#         seq_ = torch.cat([base_pos[:, None], seq], dim=1)
#         max_steps_cross = 12

#         sample = self.base_dist(base_pos, step=0).sample()
#         u, log_q_u = self.cif.q_u_given_z.sample(torch.cat([sample, cond], dim=-1))
#         log_r_u = self.cif.r_u_given_x.log_prob(u, torch.cat([seq[:, -1], cond], dim=-1))
#         self.net.vector_field.net.data_control.cond = u

#         loss = []
#         for ts in range(self.pred_len):
#             cross = min(max_steps_cross, self.pred_len - ts)
#             loss_ts = 0
#             t = sample_t(num_samples=batch_size, a=4, device=seq.device)
#             for t_diff in range(1, cross+1):
#                 t, xt, ut = self.FM.sample_location_and_conditional_flow(seq_[:, ts], seq_[:, ts + t_diff], t=t)
#                 vt = self.net.vector_field.net(t * t_diff + ts, xt)
#                 loss_ts += (vt - ut / t_diff).pow(2).mean(dim=-1)
#             loss.append(loss_ts)
        
#         loss = torch.stack(loss, dim=1)
#         log_r_u = self.cif.r_u_given_x.log_prob(u, torch.cat([seq[:, -1], cond], dim=-1))
#         loss -= (log_r_u - log_q_u)[:, None] / self.pred_len * 0.01

#         if self.hybrid_loss:
#             loss -= self.log_prob(base_pos, seq, cond)
    
#         return loss
    
    
class FlowMatchingChain_CNF_CIF_spline(FlowMatchingChain_CNF_CIF):
    def flow_matching(self, base_pos, seq, cond):
        batch_size, *_ = seq.shape
        # base positions are allways equal to zero by position normalization
        # assert base_pos.allclose(torch.zeros_like(base_pos))
        sample = self.base_dist(base_pos, step=0).sample()
        
        grid = torch.arange(self.pred_len+1, dtype=torch.float, device=seq.device)
        points = torch.cat([sample[:, None], seq], dim=1)
        
        coeffs = natural_cubic_spline_coeffs(grid, points)
        spline = NaturalCubicSpline(coeffs)
                
        loss = []
        u_orig, log_q_u = self.cif.q_u_given_z.sample(torch.cat([sample, cond], dim=-1))
        k = 1
        u = u_orig.repeat_interleave(k, dim=0)
        for i in range(self.pred_len):
            t = sample_t(num_samples=batch_size, a=4, device=seq.device)
            t += i
            index = torch.eye(batch_size, dtype=torch.bool)
            xt = spline.evaluate(t)[index]
            
            xt = xt.repeat_interleave(k, dim=0)
            noise = self.base_sigma * torch.randn_like(xt)
            ut = spline.derivative(t)[index]
            ut = ut.repeat_interleave(k, dim=0)
            t = t.repeat_interleave(k, dim=0)
            self.net.vector_field.net.data_control.cond = u
            vt = self.net.vector_field.net(t, xt + noise)
            if False:
                ut_ortho = ut.clone()
                ut_ortho[:, 0] = -ut[:, 1]
                ut_ortho[:, 1] = ut[:, 0]
                ut_ortho = ut_ortho / ut_ortho.norm(dim=-1, keepdim=True)
                noise_perpendicular = - (noise * ut_ortho).sum(dim=-1, keepdim=True) * ut_ortho
                ut = ut + noise_perpendicular
            loss.append((vt - ut).reshape(batch_size, k, -1).pow(2).mean(dim=-1).min(dim=1).values)
        loss = torch.stack(loss, dim=1)
        log_r_u = self.cif.r_u_given_x.log_prob(u_orig, torch.cat([seq[:, -1], cond], dim=-1))
        loss -= (-log_q_u + log_r_u)[:, None] / self.pred_len * 0.01
        
        if self.hybrid_loss:
            loss -= self.log_prob(base_pos, seq, cond)
        
        return loss
        
        
class FlowMatchingChain_CNF_spline(FlowMatchingChain_CNF):
    def flow_matching(self, base_pos, seq, cond):
        batch_size, *_ = seq.shape
        
        sample = self.base_dist(base_pos, step=0).sample()
        grid = torch.arange(self.pred_len+1, dtype=torch.float, device=seq.device)
        points = torch.cat([sample[:, None], seq], dim=1)
        coeffs = natural_cubic_spline_coeffs(grid, points)
        spline = NaturalCubicSpline(coeffs)
        
        loss = []
        for i in range(self.pred_len):
            t = sample_t(num_samples=batch_size, a=4, device=seq.device)
            t += i
            index = torch.eye(batch_size, dtype=torch.bool)
            xt = spline.evaluate(t)[index]
            xt += self.base_sigma * torch.randn_like(xt)
            ut = spline.derivative(t)[index]
            self.net.vector_field.net.data_control.cond = cond
            vt = self.net.vector_field.net(t, xt)
            loss.append((vt - ut).pow(2).mean(dim=-1))
        loss = torch.stack(loss, dim=1)
        
        return loss
    
    
def sample_t(num_samples, a, device=None):
    t = torch.distributions.exponential.Exponential(rate=a).sample([num_samples]).to(device)
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t