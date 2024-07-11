import os
from yacs.config import CfgNode
from typing import Dict

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from torchdyn.core import NeuralODE
from torchdyn.models import CNF, hutch_trace

from .mid import Trajectron
from .flowchain import (
    FlowChainTraj,
    FlowChain,
    CIF_step,
)


class ContinuousFlowChainTraj(FlowChainTraj):
    def _build_encoder(self, cfg: CfgNode) -> None:
        super()._build_encoder(cfg)
        
        path = f"experiments/configs/traj/FlowChain/CIF_separate_cond_v/{cfg.DATA.DATASET_NAME}/ckpt.pt"
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.registrar.load_models(ckpt['encoder'])
        
        # craete new Trajectron with loaded weights
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        self.encoder.set_environment(self.env)
        self.encoder.set_annealing_params()
        
    def _build_model_dict(self, cfg: CfgNode) -> Dict:
        model_dict = {
            "ContinuousFlowChainOnestep": FlowChain_CNF,
            "ContinuousFlowChainOnestep_CIF": FlowChain_CNF_CIF,
        }
            
        return model_dict
            

class FlowChain_CNF(FlowChain):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        self.input_size = input_size
        self.net = CNF_step(input_size,
                            n_hidden,
                            hidden_size,
                            cond_size)
                
    def forward_separate(self, seq, cond):
        seq_prev, seq_log_det_jacobians = self.net(seq[:, -1], cond, steps=list(range(self.pred_len, -1, -1)))
        return torch.flip(seq_prev, dims=(1,)), torch.flip(seq_log_det_jacobians, dims=(1,))
    
    def inverse(self, base_pos, cond):
        seq, seq_log_det_jacobian_cumsum = self.net(base_pos, cond, steps=list(range(0, self.pred_len+1)))
        seq_log_det_jacobian = calc_diff(seq_log_det_jacobian_cumsum)
        return seq, seq_log_det_jacobian_cumsum, seq_log_det_jacobian

        
class FlowChain_CNF_CIF(FlowChain):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):     
        self.input_size = input_size
        self.net = CNF_step(input_size,
                            n_hidden,
                            hidden_size,
                            cond_size)
        
        self.cif = CIF_step(self.net,
                            input_size=input_size,
                            cond_size=cond_size)
        
    def forward_separate(self, seq, cond):
        u, log_r_u = self.cif.r_u_given_x.sample(torch.cat([seq[:, -1], cond], dim=-1))
        seq_prev, seq_log_det_jacobians = self.net(seq[:, -1], u, steps=list(range(self.pred_len, -1, -1)))
        seq_prev = torch.flip(seq_prev, dims=(1,))
        seq_log_det_jacobians = torch.flip(seq_log_det_jacobians, dims=(1,))
        log_q_u = self.cif.q_u_given_z.log_prob(u, torch.cat([seq_prev[:, 0], cond], dim=-1))
        seq_log_det_jacobians -= (log_q_u - log_r_u)[:, None] / self.pred_len
        return seq_prev, seq_log_det_jacobians
    
    def inverse(self, base_pos, cond):
        u, log_q_u = self.cif.q_u_given_z.sample(torch.cat([base_pos, cond], dim=-1))
        seq, seq_log_det_jacobians_cumsum = self.net(base_pos, u, steps=list(range(0, self.pred_len+1)))
        log_r_u = self.cif.r_u_given_x.log_prob(u, torch.cat([seq[..., -1, :], cond], dim=-1))
        seq_log_det_jacobians_cumsum -= (log_r_u - log_q_u)[..., None].expand(-1, -1, self.pred_len) / torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=2)
        seq_log_det_jacobians = calc_diff(seq_log_det_jacobians_cumsum)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians
        

class CNF_step(nn.Module):
    def __init__(self, 
                 input_size,
                 n_hidden,
                 hidden_size,
                 cond_size,
                 solver='dopri5',
                 trace_estimator=None,
                 time_varying=True  # integrating reverse got minus timesteps
                 ):
        super(CNF_step, self).__init__()
        
        self.input_size = input_size
        self.cond_size = cond_size
        
        self.time_varying = time_varying
        
        dynamics = MLP(input_size,
                            n_hidden,
                            hidden_size,
                            cond_size,
                            time_varying=time_varying)
        
        if trace_estimator is None:
            trace_estimator = hutch_trace
        
        self.vector_field = CustomCNF(dynamics,
                                      trace_estimator=trace_estimator,
                                      noise_dist=MultivariateNormal(torch.zeros(2), torch.eye(2))
                                      )
        
        if solver in ['euler', 'midpoint', 'rk4']:
            self.num_step = 4
            self.ode = NeuralODE(self.vector_field,
                                 solver=solver)
            
        elif solver in ['dopri5', 'tsit5']:
            self.num_step = 2
            tolerance = 1e-2
            self.ode = NeuralODE(self.vector_field,
                                 sensitivity='adjoint',
                                 solver=solver,
                                 atol=tolerance,
                                 rtol=tolerance)
                
    def forward(self, x0, y, steps=[0, 1]):
        x0_pad = torch.nn.functional.pad(x0, (1, 0), "constant", 0)
        x0_pad_shape = x0_pad.shape
        x0_pad = x0_pad.reshape(-1, 1+self.input_size)
        self.vector_field.net.data_control.cond = y.reshape(-1, self.cond_size)
        
        indices, t_span = get_indices_t_span(steps, self.num_step)
        _, traj = self.ode(x0_pad, t_span=t_span)
        
        traj = traj[indices].transpose(0, 1).reshape([*x0_pad_shape[:-1], len(indices), x0_pad_shape[-1]])
        ldj, x1 = traj[..., 0], traj[..., 1:]
        
        return x1, ldj
    
    
def get_indices_t_span(steps, num_step):
    indices = []
    t_span = []
    for i in range(len(steps[:-1])):
        if i == 0:
            indices.append(num_step-1)
            t_span.append(torch.linspace(steps[i], steps[i+1], num_step))
        else:
            indices.append(indices[-1] + num_step-1)
            t_span.append(torch.linspace(steps[i], steps[i+1], num_step)[1:])
    t_span = torch.cat(t_span)
    
    return indices, t_span


def calc_diff(seq):
    return seq - torch.nn.functional.pad(seq[..., :-1, :], (0, 0, 1, 0), "constant", 0)
    
    
class CustomCNF(CNF):
    def __init__(self, net,
                 trace_estimator=None,
                 noise_dist=None,):
                 
        super().__init__(
            net=net,
            trace_estimator=trace_estimator,
            noise_dist=noise_dist,
        )
            
    def forward(self, t, xt, args={}):
        timesteps = t.abs()
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() in [0, 1]:
            timesteps = timesteps.repeat(xt.shape[0])
                    
        with torch.set_grad_enabled(True):
            # first dimension reserved to store the log determinant of the Jacobian
            x_in = xt[:,1:].requires_grad_(True)
            # actual score computation
            x_out = self.net(timesteps, x_in)
                
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise.to(x_in.device))
        return torch.cat([-trJ[:, None], x_out], 1) + 0*xt # `+ 0*xt` has the only purpose of connecting x[:, 0] to autograd graph
    
    
class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 n_hidden,
                 hidden_size,
                 cond_size,
                 time_varying):
        super(MLP, self).__init__()
        
        self.time_varying = time_varying
        
        self.data_control = CustomDataControl(cond_size)
        
        self.net = nn.Sequential()
        self.net.append(nn.Linear(input_size + (1 if time_varying else 0) + cond_size, hidden_size))
        self.net.append(nn.SELU())
        
        for _ in range(n_hidden):
            self.net.append(nn.Linear(hidden_size, hidden_size))
            self.net.append(nn.SELU())
            
        self.net.append(nn.Linear(hidden_size, input_size))
    
    def forward(self, t, xt):
        if self.time_varying:
            xt = torch.cat([xt, t[:, None]], 1)
        xt = self.data_control(xt)
        return self.net(xt)
    
    
class CustomDataControl(nn.Module):
    """Data-control module. Allows for datasets-control inputs at arbitrary points of the DEFunc
    """
    def __init__(self, r):
        super().__init__()
        self.u = None
        self.cond = None

    def forward(self, xt):
        return torch.cat([xt, self.cond], 1)
    
