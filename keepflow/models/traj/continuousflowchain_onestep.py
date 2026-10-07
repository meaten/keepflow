import os
from yacs.config import CfgNode
from typing import Dict
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from torchdyn.core import NeuralODE
from torchdyn.models import CNF, hutch_trace
from torchdyn.nn import Augmenter

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
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.registrar.load_models(ckpt['encoder'])
        
        # craete new Trajectron with loaded weights
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        self.encoder.set_environment(self.env)
        self.encoder.set_annealing_params()
        
        
    def _build_flow_arg(self, cfg: CfgNode) -> dict:
        args = super()._build_flow_arg(cfg)
        args['cnf_solver'] = cfg.MODEL.FLOW.CNF_SOLVER
        args['num_step'] = cfg.MODEL.FLOW.NUM_STEP
        args['atol'] = cfg.MODEL.FLOW.CNF_ATOL
        args['rtol'] = cfg.MODEL.FLOW.CNF_RTOL
        return args

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
                            cond_size,
                            **solver_kwargs(kwargs))
                
    def forward_separate(self, seq, cond):
        # seq_prev = []
        # seq_log_det_jacobians = []
        # for i in range(self.pred_len):
        #     prev, log_det_jacobian = self.net(seq[:, i], cond, steps=[i+1, i])
        #     seq_prev.append(prev)
        #     seq_log_det_jacobians.append(log_det_jacobian)
        # seq_prev = torch.cat(seq_prev, dim=1)
        # seq_log_det_jacobians = torch.cat(seq_log_det_jacobians, dim=1)
        seq_prev, seq_log_det_jacobians = self.net(seq[:, -1], cond, steps=list(range(self.pred_len, -1, -1)))
        seq_prev = torch.flip(seq_prev, [1])
        seq_log_det_jacobians = torch.flip(seq_log_det_jacobians, [1])
        return seq_prev, seq_log_det_jacobians
    
    def chain_preimages(self, seq, cond, n_draws=1):
        """Same as the CIF variant, without the latent index."""
        T = self.pred_len
        B, _, dim = seq.shape
        preimages = seq.new_zeros(B, T, T, dim)
        ldj_cum = seq.new_zeros(B, T, T)
        for n in range(T):
            z, ldj = self.net(seq[:, n], cond, steps=list(range(n + 1, -1, -1)))
            for k in range(n + 1):
                preimages[:, n, k] = z[:, n - k]
                ldj_cum[:, n, k] = ldj[:, n - k]
        return preimages, ldj_cum, None

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
                            cond_size,
                            **solver_kwargs(kwargs))
        
        self.cif = CIF_step(self.net,
                            input_size=input_size,
                            cond_size=cond_size,
                            mixture_r_u=False)
        
    def forward_separate(self, seq, cond):
        u, log_r_u = self.cif.r_u_given_x.sample(torch.cat([seq[:, -1], cond], dim=-1))
        seq_prev = []
        seq_log_det_jacobians = []
        for i in range(self.pred_len):
            prev, log_det_jacobian = self.net(seq[:, i], u, steps=[i+1, i])
            seq_prev.append(prev)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_prev = torch.cat(seq_prev, dim=1)
        seq_log_det_jacobians = torch.cat(seq_log_det_jacobians, dim=1)
        # z, _ = self.net(seq[:, -1], u, steps=[self.pred_len-1, 0])
        # log_q_u = self.cif.q_u_given_z.log_prob(u, torch.cat([z[:, 0], cond], dim=-1))
        log_q_u = self.cif.q_u_given_z.log_prob(u, torch.cat([seq_prev[:, 0], cond], dim=-1))
        seq_log_det_jacobians += (-log_q_u + log_r_u)[:, None] / self.pred_len
        return seq_prev, seq_log_det_jacobians
    
    def chain_preimages(self, seq, cond, n_draws=1):
        """Take every ground-truth position back through the whole chain.

        `log_prob_separate` inverts one segment per step and scores the result
        against a base Gaussian centred on the previous ground-truth position, so
        it can only describe the density before any update. The fast update moves
        the base Gaussian to a later observation and restarts the product of
        log-det-Jacobians there, which needs the preimage of the ground truth at
        that later step.

        The transformations do not change when an observation arrives, so neither
        do the preimages. Computing them once here lets every update be evaluated
        by re-centring the Gaussian and re-summing cached terms, with no further
        integration. That mirrors what the particle path does: the particles stay
        put and only their weights change.

        `n_draws` averages the stochastic trace estimate over independent draws.
        It reduces the variance of the metric without altering the model, which
        still uses a single draw at inference.

        Returns (preimages, ldj_cum, cif_correction) where preimages[:, n, k] is
        the preimage of seq[:, n] at step k and ldj_cum[:, n, k] the accumulated
        log-det-Jacobian of that inversion. Entries with k > n are unused.
        """
        u, log_r_u = self.cif.r_u_given_x.sample(
            torch.cat([seq[:, -1], cond], dim=-1))
        T = self.pred_len
        B, _, dim = seq.shape
        preimages = seq.new_zeros(B, T, T, dim)
        ldj_cum = seq.new_zeros(B, T, T)
        for n in range(T):
            zs, ls = [], []
            for _ in range(n_draws):
                z, ldj = self.net(seq[:, n], u, steps=list(range(n + 1, -1, -1)))
                zs.append(z); ls.append(ldj)
            z = torch.stack(zs).mean(0)
            ldj = torch.stack(ls).mean(0)
            for k in range(n + 1):
                preimages[:, n, k] = z[:, n - k]
                ldj_cum[:, n, k] = ldj[:, n - k]
        log_q_u = self.cif.q_u_given_z.log_prob(
            u, torch.cat([preimages[:, 0, 0], cond], dim=-1))
        return preimages, ldj_cum, (log_q_u - log_r_u)

    def inverse(self, base_pos, cond):
        u, log_q_u = self.cif.q_u_given_z.sample(torch.cat([base_pos, cond], dim=-1))
        seq, seq_log_det_jacobians_cumsum = self.net(base_pos, u, steps=list(range(0, self.pred_len+1)))
        log_r_u = self.cif.r_u_given_x.log_prob(u, torch.cat([seq[..., -1, :], cond], dim=-1))
        seq_log_det_jacobians_cumsum += (log_q_u - log_r_u)[..., None].expand(-1, -1, self.pred_len) / torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=2)
        seq_log_det_jacobians = calc_diff(seq_log_det_jacobians_cumsum)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians
        

def solver_kwargs(kwargs):
    """ODE solver settings from MODEL.FLOW (see ContinuousFlowChainTraj._build_flow_arg)."""
    return {k: kwargs[k] for k in ('cnf_solver', 'num_step', 'atol', 'rtol') if k in kwargs}


class CNF_step(nn.Module):
    def __init__(self, 
                 input_size,
                 n_hidden,
                 hidden_size,
                 cond_size,
                 cnf_solver='dopri5',
                 num_step=None,
                 atol=1e-2,
                 rtol=1e-2,
                 trace_estimator=None,
                 time_varying=True  # integrating reverse got minus timesteps
                 ):
        super(CNF_step, self).__init__()
        
        self.input_size = input_size
        self.cond_size = cond_size
        
        self.time_varying = time_varying
        
        dynamics = MLP(
            input_size,
            n_hidden,
            hidden_size,
            cond_size,
            time_varying=time_varying)
        
        if trace_estimator is None:
            trace_estimator = hutch_trace
            # trace_estimator = autograd_trace
            
        self.mean = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.std = nn.Parameter(torch.ones(input_size), requires_grad=False)
        
        self.vector_field = CustomCNF(
            dynamics,
            trace_estimator=trace_estimator,
            noise_dist=Normal(self.mean, self.std)
            )
        
        solver = cnf_solver
        if solver in ['euler', 'midpoint', 'rk4']:
            # NUM_STEP solver steps per unit of time, i.e. NUM_STEP + 1 grid points.
            assert num_step is not None, "fixed-step solvers need MODEL.FLOW.NUM_STEP"
            self.num_step = num_step + 1
            self.nde = NeuralODE(self.vector_field,
                                 solver=solver)
            
        elif solver in ['dopri5', 'tsit5']:
            # only the end points of each unit interval are read out
            self.num_step = 2
            self.nde = NeuralODE(self.vector_field,
                                 sensitivity='adjoint',
                                 solver=solver,
                                 atol=atol,
                                 rtol=rtol)
        else:
            raise ValueError(f"unknown CNF solver {solver}")
            
        self.nde = torch.nn.Sequential(
            Augmenter(augment_idx=1, augment_dims=1),
            self.nde,)
                
    def forward(self, x0, y, steps=[0, 1]):
        x0_shape = x0.shape
        x0 = x0.reshape(-1, self.input_size)
        self.vector_field.net.data_control.cond = y.reshape(-1, self.cond_size)
        indices, t_span = get_indices_t_span(steps, self.num_step)
        self.nde[1].t_span = t_span
        _, traj = self.nde(x0)
        self.nde[1].nfe = 0
        traj = traj[indices].transpose(0, 1)
        ldj, x1 = traj[..., 0], traj[..., 1:]
        x1_shape = x0_shape[:-1] + (len(indices), ) + x0_shape[-1:]
        x1 = x1.reshape(*x1_shape)
        ldj = ldj.reshape(*x1_shape[:-1])
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
    return seq - torch.nn.functional.pad(seq[..., :-1], (1, 0), "constant", 0)
    
    
class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 n_hidden,
                 hidden_size,
                 cond_size,
                 time_varying,):
        super(MLP, self).__init__()
        
        self.data_control = CustomDataControl(time_varying=time_varying,
                                              embedding_type='sine',
                                              SFT=False,
                                              cond_size=cond_size)
        size = input_size + cond_size + (self.data_control.time_embedding.dim() if time_varying else 0)
        self.net = nn.Sequential()
        self.net.append(nn.Linear(size, hidden_size))
        self.net.append(nn.ReLU())
        
        for _ in range(n_hidden):
            self.net.append(nn.Linear(hidden_size, hidden_size))
            self.net.append(nn.ReLU())
            
        self.net.append(nn.Linear(hidden_size, input_size))
    
    def forward(self, t, xt, args={}):
        inputs = self.data_control(t, xt)
        return self.net(inputs)
    
class time_embedding():
    def __init__(self, type='linear'):
        self.pred_len = 12
        self.type = type
    def dim(self):
        return 6 if self.type == 'sine' else 1
    def __call__(self, t):
        if self.type == 'linear':
            return t
        elif self.type == 'sine':
            return torch.cat([
                torch.sin(np.pi*t/self.pred_len),
                torch.cos(np.pi*t/self.pred_len),
                torch.sin(2*np.pi*t/self.pred_len),
                torch.cos(2*np.pi*t/self.pred_len),
                torch.sin(4*np.pi*t/self.pred_len),
                torch.cos(4*np.pi*t/self.pred_len),
                ], 1)
    
    
class CustomDataControl(nn.Module):
    """Data-control module. Allows for datasets-control inputs at arbitrary points of the DEFunc
    """
    def __init__(self, time_varying, embedding_type, SFT, cond_size):
        super().__init__()
        self.u = None
        self.cond = None
        
        self.time_varying = time_varying
        if self.time_varying:
            self.time_embedding = time_embedding(embedding_type)
            dim = self.time_embedding.dim()
            self.SFT = SFT
            if self.SFT:
                self.mult = nn.Sequential(
                    nn.Linear(dim, cond_size),
                    nn.ReLU(),
                    nn.Linear(cond_size, cond_size),
                )
                self.add = nn.Sequential(
                    nn.Linear(dim, cond_size),
                    nn.ReLU(),
                    nn.Linear(cond_size, cond_size),
                )

    def forward(self, t, xt):
        if self.time_varying:
            t = self.time_embedding(t[:, None])
            if self.SFT:
                cond = self.mult(t) * self.cond + self.add(t)
            else:
                cond = self.cond
            return torch.cat([xt, t, cond], 1)
        else:
            return torch.cat([xt, self.cond], 1)
            
    
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
        timesteps = t
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