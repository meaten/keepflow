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
    FlowChain_separate,
    FlowChain_separate_cond,
    CIF_step,
)


class ContinuousFlowChainTraj(FlowChainTraj):
    def _build_encoder(self, cfg: CfgNode) -> None:
        super()._build_encoder(cfg)
        
        # path = f"experiments/configs/traj/FlowChain/CIF_separate_cond_v_trajectron/{cfg.DATA.DATASET_NAME}/ckpt.pt"
        # if os.path.exists(path):
        #     ckpt = torch.load(path, map_location=self.device)
        #     self.registrar.load_models(ckpt['encoder'])
        
        # # craete new Trajectron with loaded weights
        # self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        # self.encoder.set_environment(self.env)
        # self.encoder.set_annealing_params()
        
    def _build_model_dict(self, cfg: CfgNode) -> Dict:
        model_dict = {
            "ContinuousFlowChain": FlowChain_CNF,
            "ContinuousFlowChain_separate": FlowChain_CNF_separate,
            "ContinuousFlowChain_separate_cond": FlowChain_CNF_separate_cond,
            "ContinuousFlowChain_CIF": FlowChain_CNF_CIF,
            "ContinuousFlowChain_CIF_separate": FlowChain_CNF_CIF_separate,
            "ContinuousFlowChain_CIF_separate_cond": FlowChain_CNF_CIF_separate_cond,
        }
            
        return model_dict
            

class FlowChain_CNF(FlowChain):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        self.net = CNF_step(input_size,
                            n_hidden,
                            hidden_size,
                            cond_size)
        
        
class FlowChain_CNF_separate(FlowChain_separate):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):        
        pred_len = kwargs['pred_len']
        self.net = nn.ModuleList(CNF_step(input_size,
                                          n_hidden,
                                          hidden_size,
                                          cond_size)
                                 for _ in range(pred_len))
        
        
class FlowChain_CNF_separate_cond(FlowChain_separate_cond):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        pred_len = kwargs['pred_len']
        self.net = nn.ModuleList(CNF_step(input_size,
                                          n_hidden,
                                          hidden_size,
                                          cond_size + i * input_size,)
                                 for i in range(pred_len))
        
        
class FlowChain_CNF_CIF(FlowChain):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):     
        bijection = CNF_step(input_size,
                             n_hidden,
                             hidden_size,
                             cond_size)
        
        self.net = CIF_step(bijection,
                            input_size=self.input_size,
                            cond_size=cond_size)
     
        
class FlowChain_CNF_CIF_separate(FlowChain_separate):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        pred_len = kwargs['pred_len']
        self.net = nn.ModuleList(CIF_step(CNF_step(input_size,
                                                   n_hidden,
                                                   hidden_size,
                                                   cond_size),
                                          input_size=self.input_size,
                                          cond_size=cond_size)
                                 for _ in range(pred_len))
     
        
class FlowChain_CNF_CIF_separate_cond(FlowChain_separate_cond):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        pred_len = kwargs['pred_len']
        self.net = nn.ModuleList(CIF_step(CNF_step(input_size,
                                                   n_hidden,
                                                   hidden_size,
                                                   cond_size + i * input_size,),
                                          input_size=self.input_size,
                                          cond_size=cond_size + i * self.input_size)
                                 for i in range(pred_len))
        

class CNF_step(nn.Module):
    def __init__(self, 
                 input_size,
                 n_hidden,
                 hidden_size,
                 cond_size,
                 solver='dopri5',
                 trace_estimator=None,
                 time_varying=True):
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
                
    def forward(self, x1, y):
        x1_pad = torch.nn.functional.pad(x1, (1, 0), "constant", 0)
        shape_x_pad = x1_pad.shape
        self.vector_field.net.data_control.cond = y.reshape(-1, self.cond_size)
        
        _, traj = self.ode(x1_pad.reshape(-1, 1+self.input_size),
                           t_span=torch.linspace(1, 0, self.num_step, device=x1_pad.device),)
        
        traj = traj[-1].reshape(shape_x_pad)
        ldj, x0 = traj[..., 0], traj[..., 1:]
        
        return x0, ldj
    
    def inverse(self, x0, y):
        x0_pad = torch.nn.functional.pad(x0, (1, 0), "constant", 0)
        shape_z_pad = x0_pad.shape
        self.vector_field.net.data_control.cond = y.reshape(-1, self.cond_size)
        
        _, traj = self.ode(x0_pad.reshape(-1, 1+self.input_size),
                           t_span=torch.linspace(0, 1, self.num_step, device=x0_pad.device),)
            
        traj = traj[-1].reshape(shape_z_pad)
        ldj, x1 = traj[..., 0], traj[..., 1:]
        
        return x1, ldj
    
    
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
    
