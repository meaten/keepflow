from yacs.config import CfgNode
from typing import Dict

import numpy as np
import scipy.stats as stats
import torch

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher

from .continuousflowchain_onestep import (
    ContinuousFlowChainTraj,
    FlowChain_CNF,
    FlowChain_CNF_CIF,
)


class FlowMatchingChainTraj(ContinuousFlowChainTraj):
    def _build_model_dict(self, cfg: CfgNode) -> Dict:
        model_dict = {
            "FlowMatchingChainOnestep": FlowMatchingChain_CNF,
            "FlowMatchingChainOnestep_CIF": FlowMatchingChain_CNF_CIF,
        }
            
        return model_dict
    
    def training_step(self, data_dict: Dict) -> Dict:
        node = self.get_node_type(data_dict)
        dist_args = self._encode(data_dict, node)
        
        gt = data_dict['gt']
        if self.dequantize:
            gt += torch.rand_like(data_dict['gt']) / 100

        base_pos = self._get_base_pos(data_dict)
        
        mask = ~torch.isnan(gt)[..., 0]
        loss = self.flow[node].flow_matching(base_pos, gt.nan_to_num(), dist_args)
        loss *= mask
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()
        
        return {'loss': loss.mean().item()}
    

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
        
    def flow_matching(self, base_pos, seq, cond):
        x0 = torch.cat([base_pos[:, None], seq[:, :-1]], dim=1)
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
        
    def flow_matching(self, base_pos, seq, cond):
        x0 = torch.cat([base_pos[:, None], seq[:, :-1]], dim=1)
        x1 = seq
        
        batch_size, *_ = seq.shape

        loss = []
        sample = self.base_dist(x0[:, 0], step=0).sample()
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
        loss -= (log_r_u - log_q_u)[:, None] / self.pred_len * 0.01

        if self.hybrid_loss:
            loss -= self.log_prob(base_pos, seq, cond)
            
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