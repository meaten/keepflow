from yacs.config import CfgNode
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from trajdata.data_structures import AgentType

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.numerics import odeint

from .mid import Trajectron
from .flowchain import FlowChainTraj
from .flowmatchingchain import MLP


class FlowChainProcessTraj(FlowChainTraj):
    def _build_encoder(self, cfg: CfgNode) -> None:
        super()._build_encoder(cfg)
        
        path = f"experiments/configs/traj/FlowChain/CIF_separate_cond_v_trajectron/{cfg.DATA.DATASET_NAME}/ckpt.pt"
        
        ckpt = torch.load(path, map_location=self.device)
        self.registrar.load_models(ckpt['encoder'])
        
        # craete new Trajectron with loaded weights
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        self.encoder.set_environment(self.env)
        self.encoder.set_annealing_params()
        
    def _build_flow(self, cfg: CfgNode):
        ### dynamics model, dz/dt
        model_dict = {
            "FlowChainProcess": MLP
        }
        
        conditioning_length = \
            self.hyperparams["enc_rnn_dim_history"] + \
            self.hyperparams["enc_rnn_dim_edge"] * self.hyperparams["edge_encoding"] + \
            self.hyperparams["enc_rnn_dim_future"] * self.hyperparams["incl_robot_node"] + 1
            
        self.flow = nn.ModuleDict()
        for node_type in self.env.NodeType:
            self.flow[node_type] = model_dict[cfg.MODEL.TYPE](
                dim_condition=conditioning_length,
            )
            
        self.var = nn.Parameter(torch.ones(2)/10, requires_grad=False)
            
        self.sigma_FM = 0.01
        self.FM = ConditionalFlowMatcher(sigma=self.sigma_FM)
    
    def predict(self, data_dict: Dict, return_prob: bool = True) -> Dict:
        if return_prob:
            return self._predict_CNF(data_dict)
        else:
            return self._predict_ODE(data_dict)
        
    def _predict_ODE(self, data_dict: Dict) -> Dict:
        node = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')
        dist_args = self._encode(data_dict, node)
        
        num_sample = 1
        
        base_pos = self._get_base_pos(data_dict)
        batch_size, dim = base_pos.shape
        
        gaussian = Normal(torch.zeros(dim).to(self.device), torch.ones(dim).to(self.device) * self.var)
        x0 = gaussian.sample(sample_shape=[batch_size * num_sample, self.pred_len])
        x0 = torch.cumsum(x0, dim=1).reshape(batch_size * num_sample * self.pred_len, -1)
        
        dist_args = dist_args[:, None, None].tile([1, num_sample, self.pred_len, 1])
        tau = torch.arange(self.pred_len)[None, None, :] / self.pred_len
        tau = tau.to(self.device).tile([batch_size, num_sample, 1])
        dist_args = torch.cat([dist_args, tau[..., None]], dim=-1)
        dist_args = dist_args.reshape([batch_size * num_sample * self.pred_len, -1])
        
        ode = ConditionalODE(self.flow[node])

        _, traj = odeint(
            lambda t, x: ode(t, x, dist_args),
            x0,
            torch.linspace(0, 1, 2, device=self.device),
            atol=1e-4,
            rtol=1e-4,
            solver="dopri5"
        )
        
        traj = traj[-1].reshape(batch_size, num_sample, self.pred_len, -1)
        
        data_dict[("pred", 0)] = traj[:, 0]
        
        return data_dict
    
    def update_from_new_obs(self, data_dict: Dict, time_step: int) -> Dict:
        return data_dict
    
    def training_step(self, data_dict: Dict) -> Dict:
        # assume all agent types are the same in this minibatch
        node = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')
        dist_args = self._encode(data_dict, node)
        
        gt = data_dict['gt']
        batch_size, pred_len, dim = gt.shape
        
        dist_args = dist_args[:, None].tile([1, pred_len, 1])
        tau = torch.arange(pred_len)[None, :] / pred_len
        tau = tau.to(self.device).tile([batch_size, 1])
        dist_args = torch.cat([dist_args, tau[..., None]], dim=-1)
        dist_args = dist_args.reshape(batch_size * pred_len, -1)
        
        x0 = torch.randn(batch_size, pred_len, dim).to(self.device)
        x0 = torch.cumsum(x0, dim=1) * self.var
        x0 = x0.reshape(batch_size * pred_len, -1)
        x1 = gt.reshape(batch_size * pred_len, -1)
        
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        
        vt = self.flow[node](torch.cat([xt, t[..., None], dist_args], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.mean().item()}
    
    def get_node_type(self, data_dict: Dict) -> AgentType:
        return str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')
    
    def get_dist_arg(self, data_dict: Dict, num_sample: int) -> torch.Tensor:
    
    
class ConditionalODE(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self, t, x, dist_args):
        timesteps = t
        while timesteps.dim() > 1:
            print(timesteps.shape)
            timesteps = timesteps[:, 0]
        if timesteps.dim() in [0, 1]:
            timesteps = timesteps.repeat(x.shape[0])
        
        x = torch.cat([x, timesteps[..., None], dist_args], dim=-1)
        
        x_out = self.net(x)
        
        return x_out