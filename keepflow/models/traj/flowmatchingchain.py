from yacs.config import CfgNode
from typing import Dict

import numpy as np
import scipy.stats as stats
import torch

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher

from .continuousflowchain import (
    ContinuousFlowChainTraj,
    FlowChain_CNF,
    FlowChain_CNF_separate,
    FlowChain_CNF_separate_cond,
    FlowChain_CNF_CIF,
    FlowChain_CNF_CIF_separate,
    FlowChain_CNF_CIF_separate_cond,
)


class FlowMatchingChainTraj(ContinuousFlowChainTraj):
    def _build_model_dict(self, cfg: CfgNode) -> Dict:
        model_dict = {
            "FlowMatchingChain": FlowMatchingChain_CNF,
            "FlowMatchingChain_separate": FlowMatchingChain_CNF_separate,
            "FlowMatchingChain_separate_cond": FlowMatchingChain_CNF_separate_cond,
            "FlowMatchingChain_CIF": FlowMatchingChain_CNF_CIF,
            "FlowMatchingChain_CIF_separate": FlowMatchingChain_CNF_CIF_separate,
            "FlowMatchingChain_CIF_separate_cond": FlowMatchingChain_CNF_CIF_separate_cond,
        }
            
        return model_dict
    
    def training_step(self, data_dict: Dict) -> Dict:
        node = self.get_node_type(data_dict)
        cond = self._encode(data_dict, node)
        
        gt = data_dict['gt']
        if self.dequantize:
            gt += torch.rand_like(data_dict['gt']) / 100

        base_pos = self._get_base_pos(data_dict)
        
        mask = ~torch.isnan(gt)[..., 0]
        loss = self.flow[node].flow_matching(base_pos, gt.nan_to_num(), cond)
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
        self.cond_size = cond_size
        self.FM = ConditionalFlowMatcher(sigma=self.base_sigma)
        
    def flow_matching(self, base_pos, seq, cond):
        x0 = torch.cat([base_pos[:, None], seq[:, :-1]], dim=1).reshape(-1, self.input_size)
        x1 = seq.reshape(-1, self.input_size)
        
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        self.net.vector_field.net.data_control.cond = cond[:, None].expand(-1, self.pred_len, -1).reshape(-1, self.cond_size)
        vt = self.net.vector_field.net(t, xt)
        loss = (vt - ut) ** 2
        loss = loss.reshape(seq.shape).mean(dim=-1)
        
        if self.hybrid_loss:
            loss -= self.log_prob(base_pos, seq, cond)
            
        return loss
    
    
class FlowMatchingChain_CNF_separate(FlowChain_CNF_separate):
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
        
        loss = []
        for i in range(x0.shape[1]):
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0[:, i], x1[:, i])
            self.net[i].vector_field.net.data_control.cond = cond
            vt = self.net[i].vector_field.net(t, xt)
            loss.append((vt - ut) ** 2)
        
        loss = torch.stack(loss, dim=1).mean(dim=-1)
        
        if self.hybrid_loss:
            loss -= self.log_prob(base_pos, seq, cond)
            
        return loss
    

class FlowMatchingChain_CNF_separate_cond(FlowChain_CNF_separate_cond):
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
        
        batch_size, pred_len, *_ = seq.shape
        seq_cond = seq.reshape(batch_size, pred_len * self.input_size)
        
        loss = []
        for i in range(x0.shape[1]):
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0[:, i], x1[:, i])
            cond_cat = torch.cat([cond, seq_cond[:, :i * self.input_size]], dim=-1)
            self.net[i].vector_field.net.data_control.cond = cond_cat
            vt = self.net[i].vector_field.net(t, xt)
            loss.append((vt - ut) ** 2)
        
        loss = torch.stack(loss, dim=1).mean(dim=-1)
        
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
        self.cond_size = cond_size
        self.FM = ConditionalFlowMatcher(sigma=self.base_sigma)
        
    def flow_matching(self, base_pos, seq, cond):
        x0 = torch.cat([base_pos[:, None], seq[:, :-1]], dim=1).reshape(-1, self.input_size)
        x1 = seq.reshape(-1, self.input_size)
        
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        
        # forward
        u, log_r_u = self.net.r_u_given_x.sample(torch.cat([x1, cond[:, None].expand(-1, self.pred_len, -1).reshape(-1, self.cond_size)], dim=-1))
        self.net.bijection.vector_field.net.data_control.cond = u
        vt = self.net.bijection.vector_field.net(t, xt)
        log_q_u = self.net.q_u_given_z.log_prob(u, torch.cat([x0, cond[:, None].expand(-1, self.pred_len, -1).reshape(-1, self.cond_size)], dim=-1))
        loss = (vt - ut).pow(2).mean(dim=-1) - (log_q_u - log_r_u) * 0.01
        
        loss = loss.reshape(seq.shape[:-1])

        if self.hybrid_loss:
            loss -= self.log_prob(base_pos, seq, cond)
            
        return loss
    
    
class FlowMatchingChain_CNF_CIF_separate(FlowChain_CNF_CIF_separate):
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
        for i in range(x0.shape[1]):
            if i == 0:
                sample = self.base_dist(x0[:, i], step=i).sample()
            else:
                sample = x0[:, i]
            
            t = sample_t(num_samples=batch_size, a=4, device=x0.device)
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0[:, i], x1[:, i], t=t)
            
            # backward
            u_b, log_q_u_b = self.net[i].q_u_given_z.sample(torch.cat([sample, cond], dim=-1))
            # u_b, log_q_u_b = self.net[i].q_u_given_z.sample(torch.cat([x0[:, i], cond[:, i], seq_cond[:, :i * self.input_size]], dim=-1))
            self.net[i].bijection.vector_field.net.data_control.cond = u_b
            vt_b = self.net[i].bijection.vector_field.net(t, xt)
            log_r_u_b = self.net[i].r_u_given_x.log_prob(u_b, torch.cat([x1[:, i], cond], dim=-1))
            loss_b = (vt_b - ut).pow(2).mean(dim=-1) - (log_r_u_b - log_q_u_b) * 0.01
            
            loss.append(loss_b)
        
        loss = torch.stack(loss, dim=1)
        
        if self.hybrid_loss:
            loss -= self.log_prob(base_pos, seq, cond)
            
        return loss
    
    
class FlowMatchingChain_CNF_CIF_separate_cond(FlowChain_CNF_CIF_separate_cond):
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
        seq_cond = seq.reshape(batch_size, self.pred_len * self.input_size)
        
        loss = []
        for i in range(x0.shape[1]):
            if i == 0:
                sample = self.base_dist(x0[:, i], step=i).sample()
            else:
                sample = x0[:, i]
            
            t = sample_t(num_samples=batch_size, a=4, device=x0.device)
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0[:, i], x1[:, i], t=t)
            
            # backward u inference
            u_b, log_q_u_b = self.net[i].q_u_given_z.sample(torch.cat([sample, cond, seq_cond[:, :i * self.input_size]], dim=-1))
            self.net[i].bijection.vector_field.net.data_control.cond = u_b
            vt_b = self.net[i].bijection.vector_field.net(t, xt)
            log_r_u_b = self.net[i].r_u_given_x.log_prob(u_b, torch.cat([x1[:, i], cond, seq_cond[:, :i * self.input_size]], dim=-1))
            loss_b = (vt_b - ut).pow(2).mean(dim=-1) - (log_r_u_b - log_q_u_b) * 0.01
            
            loss.append(loss_b)
            
        loss = torch.stack(loss, dim=1)
        
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