import sys
from yacs.config import CfgNode
from pathlib import Path
from typing import Dict
import yaml
from easydict import EasyDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from trajdata.data_structures import AgentType

from keepflow.models import ModelTemplate
from keepflow.utils import EmptyEnv

from .arflow import (
    FlowSequential,
    LinearMaskedCoupling,
    BatchNorm,
    MADE
)

from .mid import (
    get_traj_hypers,
    ModelRegistrar,
    Trajectron
)

class FlowChainTraj(ModelTemplate):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.pred_state = cfg.DATA.TRAJ.PRED_STATE
        
        with open("extern/traj/MID/configs/baseline.yaml") as f:
            config = yaml.safe_load(f)
        self.config = EasyDict(config)
        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
                    
        self.registrar = ModelRegistrar(self.save_dir, self.device)
        
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        env = EmptyEnv(cfg)
        self.env = env
        self.encoder.set_environment(env)
        self.encoder.set_annealing_params()

        model_dict = {
            "FlowChain": FlowChain,
            "FlowChain_separate": FlowChain_separate,
            "FlowChain_separate_cond": FlowChain_separate_cond,
            "FlowChain_VFlow": FlowChain_VFlow,
            "FlowChain_VFlow_separate": FlowChain_VFlow_separate,
            "FlowChain_VFlow_separate_cond": FlowChain_VFlow_separate_cond,
            "FlowChain_CIF": FlowChain_CIF,
            "FlowChain_CIF_separate": FlowChain_CIF_separate,
            "FlowChain_CIF_separate_cond": FlowChain_CIF_separate_cond
        }

        n_blocks = cfg.MODEL.FLOW.N_BLOCKS
        n_hidden = cfg.MODEL.FLOW.N_HIDDEN
        hidden_size = cfg.MODEL.FLOW.HIDDEN_SIZE
        conditioning_length = \
            self.hyperparams["enc_rnn_dim_history"] + \
            self.hyperparams["enc_rnn_dim_edge"] * self.hyperparams["edge_encoding"] + \
            self.hyperparams["enc_rnn_dim_future"] * self.hyperparams["incl_robot_node"]

        self.flow = nn.ModuleDict()
        for node_type in env.NodeType:
            output_size = 2
            self.flow[node_type] = model_dict[cfg.MODEL.TYPE](
                input_size=output_size,
                n_blocks=n_blocks,
                n_hidden=n_hidden,
                hidden_size=hidden_size,
                cond_label_size=conditioning_length,
                flow_architecture=cfg.MODEL.FLOW.ARCHITECTURE,
                pred_len=self.pred_len
            )

        self.dequantize = cfg.SOLVER.DEQUANTIZE

        decay_parameters = [
            n for n, p in self.named_parameters() if 'bias' not in n]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters],
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = optim.Adam(
            optimizer_grouped_parameters, cfg.SOLVER.LR)
        self.optimizers = [self.optimizer]

        self.ldjs_tmp = None
        self.base_prob_normalizer_tmp = None
        
    def create_batch(self, data_dict):
        agent_type = data_dict['agent_type']
        neighbor_type = data_dict['neighbor_type']
        
        neighbors = {edge_type : [] for edge_type in self.env.EdgeType}
        neighbors_edge = {edge_type : [] for edge_type in self.env.EdgeType}
        
        for i in range(len(agent_type)):
            for edge_type in neighbors.keys():
                if AgentType[edge_type[0]].value == agent_type[i]:
                    idx = torch.where(neighbor_type[i] == AgentType[edge_type[1]].value)
                else:
                    idx = []
                
                neighbors[edge_type].append(data_dict['neighbors'][i][idx])
                neighbors_edge[edge_type].append(torch.ones_like(idx[0]))
        
        batch = (data_dict["first_history_index"],  # first_history_index
                 data_dict["obs"],  # x_t
                 data_dict["gt"],  # y_t
                 data_dict["obs"],  # x_st_t
                 data_dict["gt"],  # y_st_t
                 neighbors,  # neighbors
                 neighbors_edge,  # neighbor_edge
                 data_dict["robot_traj"],
                 data_dict["map"])
        
        return batch
    
    def encode(self, data_dict: Dict, node_type):
        batch = self.create_batch(data_dict)
        encoded_x = self.encoder.get_latent(batch, node_type)
        encoded_x = encoded_x[:, None].expand(-1, self.pred_len, -1)
        return encoded_x
    
    def predict(self, data_dict: Dict, return_prob: bool = True) -> Dict:
        if return_prob:
            return self.predict_inverse_prob(data_dict)
        else:
            return self.predict_inverse_ML(data_dict)
        # return self.predict_forward(data_dict)

    def predict_inverse_prob(self, data_dict: Dict) -> Dict:
        node = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')  # assume all agnet types are the same in this minibatch
        dist_args = self.encode(data_dict, node)
    
        sample_num = 10000

        base_pos = self.get_base_pos(
            data_dict)[:, None].expand(-1, sample_num, -1).clone()
        dist_args = dist_args[:, None].expand(-1, sample_num, -1, -1)
        sampled_seq, log_prob, seq_ldjs, base_log_prob = self.flow[node].sample_with_log_prob(
            base_pos, cond=dist_args)
        
        # for update
        self.ldjs_tmp = seq_ldjs
        self.base_prob_normalizer_tmp = base_log_prob.exp().sum(dim=1)

        data_dict[("prob", 0)] = torch.cat(
            [sampled_seq, log_prob[..., None]], dim=-1)
        """
        log_prob = torch.nan_to_num(log_prob, nan=-float('inf'))
        index_ML = log_prob[..., -1].argmax(dim=1)[:, None, None]
        index_ML = index_ML.expand(data_dict["gt"].size()).unsqueeze(1)
        data_dict[("pred", 0)] = torch.gather(sampled_seq, 1, index_ML).squeeze(1)  # sample maximum likelihood
        """
        data_dict[("pred", 0)] = sampled_seq[:, -1]
        if False:
            data_dict[("gt_traj_log_prob", 0)] = self.flow.log_prob_sequential(base_pos[:, 0], data_dict["gt"], dist_args[:, 0])
        return data_dict

    def predict_inverse_ML(self, data_dict: Dict) -> Dict:
        node = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')  # assume all agnet types are the same in this minibatch
        dist_args = self.encode(data_dict, node)
        
        sample_num = 100

        base_pos = self.get_base_pos(
            data_dict)[:, None].expand(-1, sample_num, -1).clone()
        dist_args = dist_args[:, None].expand(-1, sample_num, -1, -1)
        sampled_seq, log_prob, seq_ldjs, base_log_prob = self.flow[node].sample_with_log_prob(
            base_pos, cond=dist_args)
        
        """
        log_prob = torch.nan_to_num(log_prob, nan=-float('inf'))
        index_ML = log_prob[..., -1].argmax(dim=1)[:, None, None]
        index_ML = index_ML.expand(data_dict["gt"].size()).unsqueeze(1)
        data_dict[("pred", 0)] = torch.gather(sampled_seq, 1,
                                                 index_ML).squeeze(1)  # sample maximum likelihood
        """
        data_dict[("pred", 0)] = sampled_seq[:, -1]
        if torch.sum(torch.isnan(data_dict[('pred', 0)])):
            data_dict[("pred", 0)] = torch.where(torch.isnan(data_dict[("pred", 0)]),
                                                    data_dict['obs'][:, 0, None, 2:4].expand(data_dict[("pred", 0)].size()),
                                                    data_dict[('pred', 0)])
        return data_dict

    def predict_from_new_obs(self, data_dict: Dict, time_step: int) -> Dict:
        node = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')  # assume all agnet types are the same in this minibatch
        assert 0 < time_step and time_step < self.pred_len, f"time_step for update need to be 0~{self.pred_len-1}, got {time_step}"
        assert ("prob", 0) in data_dict.keys(), "Initial prediction needed before updating"
        
        data_dict[("prob", time_step)] = data_dict[(
            "prob", 0)][:, :, time_step:].clone()
            
        base_log_prob = self.flow[node].base_dist(
            data_dict["gt"][:, time_step-1], step=time_step).log_prob(data_dict[("prob", 0)][:, :, time_step-1, :2])
        base_log_prob = torch.sum(base_log_prob, dim=-1)
        base_prob = base_log_prob.exp()
        base_log_prob = (base_prob / base_prob.sum(dim=1) * self.base_prob_normalizer_tmp).log()
        log_prob = base_log_prob[:, :, None] + \
            torch.cumsum(self.ldjs_tmp[:, :, time_step:], dim=0) / torch.cumsum(torch.ones_like(self.ldjs_tmp[:, :, time_step:]), dim=0) 
        data_dict[("prob", time_step)][..., -1] = log_prob

        index_ML = data_dict[("prob", time_step)][:,:, -1, -1].argmax(dim=1)[:, None, None]
        index_ML = index_ML.expand(
            data_dict["gt"][:, time_step:].size()).unsqueeze(1)
        data_dict[("pred", time_step)] = torch.gather(data_dict[(
            "prob", time_step)][..., :2], 1, index_ML).squeeze(1)  # sample maximum likelihood
        if False:
            dist_args = self.encoder(data_dict)
            data_dict[("gt_traj_log_prob", time_step)] = self.flow.log_prob_sequential(data_dict["gt"][:, time_step-1],
                                                                                       data_dict["gt"],
                                                                                       dist_args,
                                                                                       n_step=self.pred_len-time_step-1)
        return data_dict
    
    """
    def predict_forward(self, data_dict: Dict) -> Dict:
        node = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')  # assume all agnet types are the same in this minibatch
        # not for *_cond models
        data_dict = self.predict_inverse(data_dict)  # for data_dict["pred"]
        dist_args = self.encode(data_dict, node)

        sample_num_per_dim = 200
        batch_size = data_dict["obs"][-1].size()[0]
        base_pos = self.get_base_pos(data_dict)[
            :, None].expand(-1, sample_num_per_dim ** self.output_size, -1).clone()
        dist_args = dist_args[:, None].expand(
            -1, sample_num_per_dim ** self.output_size, -1, -1)
        seq = torch.cartesian_prod(torch.linspace(-3, 3, steps=sample_num_per_dim),
                                   torch.linspace(-3, 3, steps=sample_num_per_dim))[None, :, None].expand(self.pred_len, -1, batch_size, -1).to(self.device)

        log_prob = self.flow.log_prob_sequential(base_pos, seq, cond=dist_args)

        data_dict[("prob", 0)] = torch.cat(
            [seq, log_prob[..., None]], dim=-1)
        return data_dict
    """
    
    def update(self, data_dict: Dict) -> Dict:
        node = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')  # assume all agnet types are the same in this minibatch
        dist_args = self.encode(data_dict, node)
        
        gt = data_dict['gt']
        if self.dequantize:
            gt += torch.rand_like(data_dict['gt']) / 100

        base_pos = self.get_base_pos(data_dict)
        
        mask = ~torch.isnan(gt)[..., 0]
        loss = -self.flow[node].log_prob(base_pos, gt.nan_to_num(), dist_args)
        loss *= mask
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()
        
        return {'loss': loss.mean().item()}

    def get_base_pos(self, data_dict: Dict) -> torch.Tensor:
        # assume x_st contains {'position', 'velocity', 'acceleration'} info
        if self.pred_state == 'state_p':
            return data_dict["obs"][:, -1, :2]
        elif self.pred_state == 'state_v':
            return data_dict["obs"][:, -1, 2:4]
        else:
            raise ValueError
        
    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.model_path

        ckpt = {
            'epoch': epoch,
            'encoder': self.registrar.model_dict,
            'flow': self.flow.state_dict(),
            'optim_state': self.optimizer.state_dict()
        }

        torch.save(ckpt, path)

    def load(self, path: Path=None) -> int:
        if path is None:
            path = self.model_path
        
        ckpt = torch.load(path, map_location=self.device)
        self.registrar.load_models(ckpt['encoder'])
        
        # craete new Trajectron with loaded weights
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        self.encoder.set_environment(self.env)
        self.encoder.set_annealing_params()
        
        self.flow.load_state_dict(ckpt['flow'])
        
        self.optimizer.load_state_dict(ckpt['optim_state'])
        epoch = ckpt["epoch"]
                    
        return epoch
    
    
class FlowChain(nn.Module):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 **kwargs):
        super().__init__()

        self.flow_dict = {
            'realNVP': create_RealNVP_step,
            'MAF': create_MAF_step,
        }

        self.build_net(n_blocks, input_size, hidden_size,
                       n_hidden, cond_label_size, **kwargs)
        pred_len = kwargs['pred_len']
        self.var = nn.Parameter(torch.ones(
            pred_len, self.input_size)/10, requires_grad=True)

    def base_dist(self, base_pos, step=0):
        return Normal(base_pos, torch.clamp(self.var[step], min=1e-4))

    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size, **kwargs):
        self.input_size = input_size
        flow_key = kwargs['flow_architecture']
        self.net = self.flow_dict[flow_key](n_blocks=n_blocks,
                                            input_size=self.input_size,
                                            hidden_size=hidden_size,
                                            n_hidden=n_hidden,
                                            cond_label_size=cond_label_size)

    # maybe need debugging
    def forward_sequential(self, seq, cond):
        x = seq[:, -1:].detach().clone()
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[1]-1, -1, -1):
            x, log_det_jacobian = self.net(x, cond[:, step:])
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x = torch.cat([seq[:, step-1][:, None], x], dim=1)
                seq_log_det_jacobians_cumsum = torch.cat([torch.zeros(seq_log_det_jacobians_cumsum.shape[[0,2]])[:, None],
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_u = x
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=1)
        return seq_u, seq_log_det_jacobians_cumsum
    

    def forward_separate(self, seq, cond):
        seq_log_det_jacobians = 0
        seq_u, log_det_jacobian = self.net(seq, cond)
        seq_log_det_jacobians += log_det_jacobian
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        for step in range(cond.shape[2]):
            u, log_det_jacobian = self.net.inverse(u, cond[:, :, step])
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq, dim=2)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=2)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians

    def log_prob(self, base_pos, x, cond):
        return self.log_prob_separate(base_pos, x, cond)
        #return self.log_prob_sequential(base_pos, x, cond)

    def log_prob_sequential(self, base_pos, x, cond, n_step=None):
        u, seq_ldjs_cumsum = self.forward_sequential(x, cond, n_step)
        base_pos = base_pos[:, None]
        base_log_prob = torch.sum(self.base_dist(base_pos).log_prob(u), dim=-1)
        return base_log_prob + seq_ldjs_cumsum

    def log_prob_separate(self, base_pos, x, cond):
        u, seq_ldjs = self.forward_separate(x, cond)
        base_pos = torch.cat([base_pos[:, None], x[:, :-1]], dim=1)
        base_log_prob = torch.sum(self.base_dist(
            base_pos, step=list(range(base_pos.shape[1]))).log_prob(u), dim=-1)
        return base_log_prob + seq_ldjs

    def sample(self, base_pos, cond):
        u = self.base_dist(base_pos).sample()
        sample, _ = self.inverse(u, cond)
        return sample

    def sample_with_log_prob(self, base_pos, cond):
        u = self.base_dist(base_pos).sample()
        sample, seq_ldjs_cumsum, seq_ldjs = self.inverse(u, cond)
        base_log_prob = torch.sum(self.base_dist(
            base_pos).log_prob(u), dim=-1)[..., None]
        return sample, base_log_prob + seq_ldjs_cumsum, seq_ldjs, base_log_prob


class FlowChain_separate(FlowChain):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size, **kwargs):
        self.input_size = input_size
        flow_key = kwargs['flow_architecture']
        pred_len = kwargs['pred_len']
        self.net = nn.ModuleList([self.flow_dict[flow_key](n_blocks=n_blocks,
                                                           input_size=self.input_size,
                                                           hidden_size=hidden_size,
                                                           n_hidden=n_hidden,
                                                           cond_label_size=cond_label_size)
                                  for _ in range(pred_len)])

    # maybe need debugging
    def forward_sequential(self, seq, cond):
        x = seq[:, -1:].detach().clone()
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[1]-1, -1, -1):
            x, log_det_jacobian = self.net[step](x, cond[:, step:])
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x = torch.cat([seq[:, step-1][:, None], x], dim=1)
                seq_log_det_jacobians_cumsum = torch.cat([torch.zeros(seq_log_det_jacobians_cumsum.shape[[0,2]])[:, None],
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_u = x
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=1)
        return seq_u, seq_log_det_jacobians_cumsum
    

    def forward_separate(self, seq, cond):
        seq_u = []
        seq_log_det_jacobians = []
        for step in range(seq.shape[1]):
            u, log_det_jacobian = self.net[step](seq[:, step], cond[:, step])
            seq_u.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_u = torch.stack(seq_u, dim=1)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians, dim=1)
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        for step in range(cond.shape[2]):
            u, log_det_jacobian = self.net[step].inverse(u, cond[:, :, step])
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq, dim=2)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=2)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians


class FlowChain_separate_cond(FlowChain_separate):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size, **kwargs):
        self.input_size = input_size
        flow_key = kwargs['flow_architecture']
        pred_len = kwargs['pred_len']
        # increase cond_label_size for conditioning trajectory
        self.net = nn.ModuleList([self.flow_dict[flow_key](n_blocks=n_blocks,
                                                           input_size=self.input_size,
                                                           hidden_size=hidden_size,
                                                           n_hidden=n_hidden,
                                                           cond_label_size=cond_label_size + i * self.input_size)
                                  for i in range(pred_len)])
    
    def forward_sequential(self, seq, cond, n_step=None):
        x = seq[:, -1:].detach().clone()
        batch_size, pred_len, *_ = seq.shape
        seq_cond = seq.reshape(batch_size, pred_len * self.input_size)[:, None]
        seq_log_det_jacobians_cumsum = 0
        
        start = len(self.net) - 1
        stop = -1 if n_step is None else start - n_step
        for step in range(start, stop, -1):
            cond_cat = torch.cat([cond[:, step:], 
                                seq_cond[..., :step * self.input_size].expand(-1, pred_len - step, -1)],
                                dim=-1)        
            x, log_det_jacobian = self.net[step](x, cond_cat)
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x = torch.cat([seq[:, step-1:step], x], dim=1)
                seq_log_det_jacobians_cumsum = torch.cat([torch.zeros_like(seq_log_det_jacobians_cumsum[:, 0:1]),
                                                   seq_log_det_jacobians_cumsum], dim=1)
        seq_u = x
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=1)
        return seq_u, seq_log_det_jacobians_cumsum
    

    def forward_separate(self, seq, cond):
        batch_size, pred_len, *_ = seq.shape
        seq_cond = seq.reshape(batch_size, pred_len * self.input_size)
        seq_u = []
        seq_log_det_jacobians = []
        for step in range(seq.shape[1]):
            cond_cat = torch.cat(
                [cond[:, step], seq_cond[:, :step * self.input_size]], dim=-1)
            u, log_det_jacobian = self.net[step](seq[:, step], cond_cat)
            seq_u.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_u = torch.stack(seq_u, dim=1)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians, dim=1)
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        batch_size, n_sample, _ = u.shape
        for step in range(cond.shape[2]):
            if not step == 0:
                seq_cond = torch.stack(seq, dim=2)
                seq_cond = seq_cond.reshape(batch_size, n_sample, -1)
                cond_cat = torch.cat([cond[:, :, step], seq_cond], dim=2)
                u, log_det_jacobian = self.net[step].inverse(u, cond_cat)
            else:
                u, log_det_jacobian = self.net[step].inverse(
                    u, cond[:, :, step])
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq, dim=2)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=2)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians


def create_RealNVP_step(n_blocks,
                        input_size,
                        hidden_size,
                        n_hidden,
                        cond_label_size=None,
                        batch_norm=True):

    modules = []
    mask = torch.arange(input_size).float() % 2
    for i in range(n_blocks):
        modules += [
            LinearMaskedCoupling(
                input_size, hidden_size, n_hidden, mask, cond_label_size
            )
        ]
        mask = 1 - mask
        modules += batch_norm * [BatchNorm(input_size)]

    return FlowSequential(*modules)


def create_MAF_step(n_blocks,
                    input_size,
                    hidden_size,
                    n_hidden,
                    cond_label_size=None,
                    activation="ReLU",
                    input_order="sequential",
                    batch_norm=True):

    modules = []
    input_degrees = None
    for i in range(n_blocks):
        modules += [
            MADE(
                input_size,
                hidden_size,
                n_hidden,
                cond_label_size,
                activation,
                input_order,
                input_degrees,
            )
        ]
        input_degrees = modules[-1].input_degrees.flip(0)
        modules += batch_norm * [BatchNorm(input_size)]

    return FlowSequential(*modules)


class Augment_VFlow(nn.Module):
    def __init__(self, input_size, aug_size):
        super(Augment_VFlow, self).__init__()

        self.input_size = input_size
        self.aug_size = aug_size

        self.register_buffer("aug_dist_mean", torch.zeros(self.aug_size))
        self.register_buffer("aug_dist_var", torch.ones(self.aug_size))

    @property
    def aug_dist(self):
        return Normal(self.aug_dist_mean, self.aug_dist_var)

    def split_z(self, z):
        split_proportions = (self.input_size, self.aug_size)
        return torch.split(z, split_proportions, dim=-1)

    def forward(self, x):
        e = self.aug_dist.sample(sample_shape=x.shape[:-1])
        log_det_jacobian = -torch.sum(self.aug_dist.log_prob(e), dim=-1)
        z = torch.cat([x, e], dim=-1)
        return z, log_det_jacobian

    def split(self, z):
        x, e = self.split_z(z)
        log_det_jacobian = torch.sum(self.aug_dist.log_prob(e), dim=-1)
        return x, log_det_jacobian


class FlowChain_VFlow(FlowChain):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 aug_size: int = 4):
        super().__init__(input_size,
                         n_blocks,
                         hidden_size,
                         n_hidden,
                         cond_label_size)

        self.input_size = input_size
        self.aug_size = aug_size
        self.flow_dim = input_size + aug_size

        self.augment = Augment_VFlow(input_size=self.input_size,
                                     aug_size=self.aug_size)

        self.net = create_RealNVP_step(n_blocks=n_blocks,
                                       input_size=self.flow_dim,
                                       hidden_size=hidden_size,
                                       n_hidden=n_hidden,
                                       cond_label_size=cond_label_size)
        # self.net = create_MAF_step(n_blocks=n_blocks,
        #                                input_size=self.flow_dim,
        #                                hidden_size=hidden_size,
        #                                n_hidden=n_hidden,
        #                                cond_label_size=cond_label_size)

    def forward_sequential(self, seq, cond):
        seq, log_det_jacobian_aug = self.augment(seq)
        log_det_jacobian_aug = log_det_jacobian_aug[-1:]
        seq_u, seq_log_det_jacobians_cumsum = super().forward_sequential(seq, cond)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq_u, log_det_jacobian = self.augment.split(seq_u)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        return seq_u, seq_log_det_jacobians_cumsum

    def forward_separate(self, seq, cond):
        seq, seq_log_det_jacobians_aug = self.augment(seq)
        seq_u, seq_log_det_jacobians = super().forward_separate(seq, cond)
        seq_log_det_jacobians += seq_log_det_jacobians_aug
        seq_u, log_det_jacobian = self.augment.split(seq_u)
        seq_log_det_jacobians += log_det_jacobian
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        u, log_det_jacobian_aug = self.augment(u)
        seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians = super().inverse(u, cond)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq, log_det_jacobian = self.augment.split(seq)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        # TODO: need to consider augmentation gap in 'seq_log_det_jacobians'
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians


class FlowChain_VFlow_separate(FlowChain_separate):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 aug_size: int = 4,
                 pred_len=8):
        super().__init__(input_size,
                         n_blocks,
                         hidden_size,
                         n_hidden,
                         cond_label_size)

        self.input_size = input_size
        self.aug_size = aug_size
        self.flow_dim = input_size + aug_size

        self.augment = Augment_VFlow(input_size=self.input_size,
                                     aug_size=self.aug_size)

        self.net = nn.ModuleList([create_RealNVP_step(n_blocks=n_blocks,
                                                      input_size=self.flow_dim,
                                                      hidden_size=hidden_size,
                                                      n_hidden=n_hidden,
                                                      cond_label_size=cond_label_size)
                                  for _ in range(pred_len)])
        # self.net = nn.ModuleList([create_MAF_step(n_blocks=n_blocks,
        #                                input_size=self.flow_dim,
        #                                hidden_size=hidden_size,
        #                                n_hidden=n_hidden,
        #                                cond_label_size=cond_label_size)
        #                           for _ in range(pred_len)])

    def forward_sequential(self, seq, cond):
        seq, log_det_jacobian_aug = self.augment(seq)
        log_det_jacobian_aug = log_det_jacobian_aug[-1:]
        seq_u, seq_log_det_jacobians_cumsum = super().forward_sequential(seq, cond)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq_u, log_det_jacobian = self.augment.split(seq_u)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        return seq_u, seq_log_det_jacobians_cumsum

    def forward_separate(self, seq, cond):
        seq, seq_log_det_jacobians_aug = self.augment(seq)
        seq_u, seq_log_det_jacobians = super().forward_separate(seq, cond)
        seq_log_det_jacobians += seq_log_det_jacobians_aug
        seq_u, log_det_jacobian = self.augment.split(seq_u)
        seq_log_det_jacobians += log_det_jacobian
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        u, log_det_jacobian_aug = self.augment(u)
        seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians = super().inverse(u, cond)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq, log_det_jacobian = self.augment.split(seq)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        # TODO: need to consider augmentation gap in 'seq_log_det_jacobians'
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians


class FlowChain_VFlow_separate_cond(FlowChain_separate_cond):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 aug_size: int = 4,
                 pred_len=8):
        super().__init__(input_size,
                         n_blocks,
                         hidden_size,
                         n_hidden,
                         cond_label_size)

        self.input_size = input_size
        self.aug_size = aug_size
        self.flow_dim = input_size + aug_size

        self.augment = Augment_VFlow(input_size=self.input_size,
                                     aug_size=self.aug_size)

        # increase cond_label_size for conditioning trajectory
        self.net = nn.ModuleList([create_RealNVP_step(n_blocks=n_blocks,
                                                      input_size=self.flow_dim,
                                                      hidden_size=hidden_size,
                                                      n_hidden=n_hidden,
                                                      cond_label_size=cond_label_size + i * self.input_size)
                                  for i in range(pred_len)])

    def forward_sequential(self, seq, cond):
        x = seq[-1:].detach().clone()
        x, log_det_jacobian_aug = self.augment(x)
        pred_len, batch_size, *_ = seq.shape
        seq_cond = seq.transpose(0, 1).reshape(
            batch_size, pred_len * self.input_size)[None]
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            cond_cat = torch.cat([cond[step:],
                                  seq_cond[..., :step * self.input_size].expand(pred_len - step, -1, -1)],
                                 dim=-1)
            x, log_det_jacobian = self.net[step](x, cond_cat)
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x_, log_det_jacobian = self.augment(seq[step-1][None])
                x = torch.cat([x_, x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([log_det_jacobian,
                                                          seq_log_det_jacobians_cumsum], dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(
            torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq_u, log_det_jacobian = self.augment.split(x)
        seq_log_det_jacobians_cumsum += log_det_jacobian

        return seq_u, seq_log_det_jacobians_cumsum

    def forward_separate(self, seq, cond):
        pred_len, batch_size, *_ = seq.shape
        seq_cond = seq.transpose(0, 1).reshape(
            batch_size, pred_len * self.input_size)
        seq, log_det_jacobians_aug = self.augment(seq)
        seq_u = []
        seq_log_det_jacobians = []
        for step in range(seq.shape[0]):
            cond_cat = torch.cat(
                [cond[step], seq_cond[..., :step * self.input_size]], dim=-1)
            u, log_det_jacobian = self.net[step](seq[step], cond_cat)
            seq_u.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_u = torch.stack(seq_u)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians += log_det_jacobians_aug
        seq_u, log_det_jacobian = self.augment.split(seq_u)
        seq_log_det_jacobians += log_det_jacobian
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_c = []
        seq_log_det_jacobians = []
        n_sample, batch_size, _ = u.shape
        u, log_det_jacobian_aug = self.augment(u)
        for step in range(cond.shape[0]):
            if not step == 0:
                seq_cond = torch.stack(seq_c).transpose(0, 1).transpose(1, 2)
                seq_cond = seq_cond.reshape(n_sample, batch_size, -1)
                cond_cat = torch.cat([cond[step], seq_cond], dim=-1)
                u, log_det_jacobian = self.net[step].inverse(u, cond_cat)
            else:
                u, log_det_jacobian = self.net[step].inverse(u, cond[step])
            seq.append(u)
            seq_c.append(u[..., :self.input_size])
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(
            torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq, log_det_jacobian = self.augment.split(seq)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        # TODO: need to consider augmentation gap in 'seq_log_det_jacobians'
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians


class DiagonalGaussianConditionalDensity(nn.Module):
    def __init__(self,
                 cond_dim,
                 out_dim) -> None:
        super().__init__()
        self.shift_net = get_mlp(
            cond_dim, [cond_dim * 2] * 2, out_dim, nn.ReLU)

        self.log_scale_net = get_mlp(
            cond_dim, [cond_dim * 2] * 2, out_dim, nn.ReLU)

    def log_prob(self, z, cond_inputs):
        means, log_stddevs = self._means_and_log_stddevs(cond_inputs)
        return diagonal_gaussian_log_prob(z, means, log_stddevs)

    def sample(self, cond_inputs):
        means, log_stddevs = self._means_and_log_stddevs(cond_inputs)
        return diagonal_gaussian_sample(means, log_stddevs)

    def _means_and_log_stddevs(self, cond_inputs):
        return self.shift_net(cond_inputs), self.log_scale_net(cond_inputs)


def diagonal_gaussian_log_prob(w, means, log_stddevs):
    assert means.shape == log_stddevs.shape == w.shape

    vars = torch.exp(log_stddevs)**2

    *_, dim = w.shape

    const_term = -.5*dim*np.log(2*np.pi)
    log_det_terms = -torch.sum(log_stddevs, dim=-1)
    product_terms = -.5*torch.sum((w - means)**2 / vars, dim=-1)

    return const_term + log_det_terms + product_terms


def diagonal_gaussian_sample(means, log_stddevs):
    epsilon = torch.randn_like(means)
    samples = torch.exp(log_stddevs)*epsilon + means

    log_probs = diagonal_gaussian_log_prob(samples, means, log_stddevs)

    return samples, log_probs


def get_mlp(
        num_input_channels,
        hidden_channels,
        num_output_channels,
        activation,
        log_softmax_outputs=False):
    layers = []
    prev_num_hidden_channels = num_input_channels
    for num_hidden_channels in hidden_channels:
        layers.append(nn.Linear(prev_num_hidden_channels, num_hidden_channels))
        layers.append(activation())
        prev_num_hidden_channels = num_hidden_channels
    layers.append(nn.Linear(prev_num_hidden_channels, num_output_channels))

    if log_softmax_outputs:
        layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


class CIF_step(nn.Module):
    def __init__(self,
                 bijection,
                 input_size,
                 cond_label_size=None) -> None:
        super().__init__()
        self.bijection = bijection
        self.cond_label_size = cond_label_size
        self.r_u_given_z = DiagonalGaussianConditionalDensity(
            input_size + cond_label_size, cond_label_size)
        self.q_u_given_w = DiagonalGaussianConditionalDensity(
            input_size + cond_label_size, cond_label_size)

    def forward(self, z, y):
        u, log_r_u = self.r_u_given_z.sample(torch.cat([z, y], dim=-1))
        w, log_det_jac = self.bijection.forward(z, u)
        log_q_u = self.q_u_given_w.log_prob(u, torch.cat([w, y], dim=-1))

        return w, log_det_jac + log_q_u - log_r_u

    def inverse(self, w, y):
        u, log_q_u = self.q_u_given_w.sample(torch.cat([w, y], dim=-1))
        z, log_det_jac = self.bijection.inverse(w, u)
        log_r_u = self.r_u_given_z.log_prob(u, torch.cat([z, y], dim=-1))

        return z, log_det_jac + log_q_u - log_r_u


class FlowChain_CIF(FlowChain):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size, **kwargs):
        self.input_size = input_size
        flow_key = kwargs['flow_architecture']
        bijection = self.flow_dict[flow_key](n_blocks=n_blocks,
                                             input_size=self.input_size,
                                             hidden_size=hidden_size,
                                             n_hidden=n_hidden,
                                             cond_label_size=cond_label_size)

        self.net = CIF_step(bijection,
                            input_size=self.input_size,
                            cond_label_size=cond_label_size)


class FlowChain_CIF_separate(FlowChain_separate):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size, **kwargs):
        self.input_size = input_size
        flow_key = kwargs['flow_architecture']
        pred_len = kwargs['pred_len']
        self.net = nn.ModuleList([CIF_step(self.flow_dict[flow_key](n_blocks=n_blocks,
                                                                    input_size=self.input_size,
                                                                    hidden_size=hidden_size,
                                                                    n_hidden=n_hidden,
                                                                    cond_label_size=cond_label_size),
                                           input_size,
                                           cond_label_size)
                                  for _ in range(pred_len)])


class FlowChain_CIF_separate_cond(FlowChain_separate_cond):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size, **kwargs):
        self.input_size = input_size
        flow_key = kwargs['flow_architecture']
        pred_len = kwargs['pred_len']
        self.net = nn.ModuleList([CIF_step(self.flow_dict[flow_key](n_blocks=n_blocks,
                                                                    input_size=self.input_size,
                                                                    hidden_size=hidden_size,
                                                                    n_hidden=n_hidden,
                                                                    cond_label_size=cond_label_size + i * self.input_size),
                                           input_size,
                                           cond_label_size + i * self.input_size)
                                  for i in range(pred_len)])
