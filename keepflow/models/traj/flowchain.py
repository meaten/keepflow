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

from .mid import (
    get_traj_hypers,
    ModelRegistrar,
    Trajectron
)

from .arflow import (
    FlowSequential,
    LinearMaskedCoupling,
    BatchNorm,
    MADE
)


class FlowChainTraj(ModelTemplate):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        
        self._build_encoder(cfg)
        self._build_flow(cfg)
        self._build_optimizer(cfg)
        
        self.ldjs_tmp = None
        self.base_prob_normalizer_tmp = None
        
    def _build_encoder(self, cfg: CfgNode) -> None:
        ### Trajectron Encoder
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

    def _build_model_dict(self, cfg: CfgNode) -> Dict:
        model_dict = {
            "FlowChain": FlowChain,
            "FlowChain_separate": FlowChain_separate,
            "FlowChain_separate_cond": FlowChain_separate_cond,
            "FlowChain_CIF": FlowChain_CIF,
            "FlowChain_CIF_separate": FlowChain_CIF_separate,
            "FlowChain_CIF_separate_cond": FlowChain_CIF_separate_cond
        }
        
        return model_dict

    def _build_flow(self, cfg: CfgNode):
        ### FlowChain Model
        model_dict = self._build_model_dict(cfg)
        
        n_blocks = cfg.MODEL.FLOW.N_BLOCKS
        n_hidden = cfg.MODEL.FLOW.N_HIDDEN
        hidden_size = cfg.MODEL.FLOW.HIDDEN_SIZE

        dim_encoder = \
            self.hyperparams["enc_rnn_dim_history"] + \
            self.hyperparams["enc_rnn_dim_edge"] * self.hyperparams["edge_encoding"] + \
            self.hyperparams["enc_rnn_dim_future"] * self.hyperparams["incl_robot_node"]
            
        self.cond_shaper = nn.Linear(dim_encoder, cfg.MODEL.FLOW.COND_SIZE)
        
        self.flow = nn.ModuleDict()
        for node_type in self.env.NodeType:
            output_size = 2
            self.flow[node_type] = model_dict[cfg.MODEL.TYPE](
                input_size=output_size,
                n_blocks=n_blocks,
                n_hidden=n_hidden,
                hidden_size=hidden_size,
                cond_size=cfg.MODEL.FLOW.COND_SIZE,
                flow_architecture=cfg.MODEL.FLOW.ARCHITECTURE,
                pred_len=self.pred_len,
                base_sigma=cfg.MODEL.FLOW.BASE_SIGMA,
            )
    
    def _build_optimizer(self, cfg: CfgNode) -> None:
        self.dequantize = cfg.SOLVER.DEQUANTIZE

        decay_parameters = [
            # n for n, p in self.flow.named_parameters() if 'bias' not in n]
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
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters, cfg.SOLVER.LR)
        self.optimizers = [self.optimizer]

    def _create_batch(self, data_dict):
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
    
    def get_node_type(self, data_dict: Dict) -> AgentType:
        # assume all agnet types are the same in this minibatch
        return str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')
    
    def _encode(self, data_dict: Dict, node_type):
        batch = self._create_batch(data_dict)
        encoded_x = self.encoder.get_latent(batch, node_type)
        cond = self.cond_shaper(encoded_x)
        return cond
    
    def predict(self, data_dict: Dict, return_prob: bool = True) -> Dict:
        if return_prob:
            return self._predict_inverse_prob(data_dict)
        else:
            return self._predict_inverse(data_dict)
        # return self.predict_forward(data_dict)

    def _predict_inverse_prob(self, data_dict: Dict) -> Dict:
        node = self.get_node_type(data_dict)
        cond = self._encode(data_dict, node)
    
        sample_num = 10000
        
        base_pos = self._get_base_pos(data_dict)
        sampled_seq, log_prob, seq_ldjs, base_log_prob = self.flow[node].sample_with_log_prob(
            base_pos[:, None].expand(-1, sample_num, -1),
            cond=cond[:, None].expand(-1, sample_num, -1))
        
        # for update
        self.ldjs_tmp = seq_ldjs
        self.base_prob_normalizer_tmp = base_log_prob.exp().sum(dim=1)

        data_dict[("prob", 0)] = torch.cat(
            [sampled_seq, log_prob[..., None]], dim=-1)

        data_dict[("pred", 0)] = sampled_seq[:, -1]
            
        return data_dict

    def _predict_inverse(self, data_dict: Dict) -> Dict:
        node = self.get_node_type(data_dict)
        cond = self._encode(data_dict, node)
        
        sample_num = 1

        base_pos = self._get_base_pos(data_dict)
        sampled_seq, log_prob, seq_ldjs, base_log_prob = self.flow[node].sample_with_log_prob(
            base_pos[:, None].expand(-1, sample_num, -1),
            cond=cond[:, None].expand(-1, sample_num, -1))
        
        data_dict[("pred", 0)] = sampled_seq[:, -1]
        if torch.sum(torch.isnan(data_dict[('pred', 0)])):
            data_dict[("pred", 0)] = torch.where(torch.isnan(data_dict[("pred", 0)]),
                                                    data_dict['obs'][:, 0, None, 2:4].expand(data_dict[("pred", 0)].size()),
                                                    data_dict[('pred', 0)])
        return data_dict

    def update_from_new_obs(self, data_dict: Dict, time_step: int) -> Dict:
        node = self.get_node_type(data_dict)
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

        return data_dict
    
    def training_step(self, data_dict: Dict) -> Dict:
        node = self.get_node_type(data_dict)
        cond = self._encode(data_dict, node)
        
        gt = data_dict['gt']
        if self.dequantize:
            gt += torch.rand_like(data_dict['gt']) / 100

        base_pos = self._get_base_pos(data_dict)
        
        mask = ~torch.isnan(gt)[..., 0]
        loss = -self.flow[node].log_prob(base_pos, gt.nan_to_num(), cond)
        loss *= mask
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()
        
        return {'loss': loss.mean().item()}

    def _get_base_pos(self, data_dict: Dict) -> torch.Tensor:
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
            'cond_shaper': self.cond_shaper.state_dict(),
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
        
        self.cond_shaper.load_state_dict(ckpt['cond_shaper'])
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
                 cond_size: int,
                 **kwargs):
        super().__init__()
        
        self.input_size = input_size
        self.pred_len = kwargs['pred_len']
        self.base_sigma = kwargs['base_sigma']
        self.logvar = nn.Parameter((torch.ones(self.pred_len, input_size) * self.base_sigma).log(),
                                requires_grad=True)
        
        self.flow_dict = {
            'realNVP': create_RealNVP_step,
            'MAF': create_MAF_step,
        }
        
        self.build_net(n_blocks, input_size, hidden_size,
                       n_hidden, cond_size, **kwargs)

    def base_dist(self, base_pos, step=0):
        return Normal(base_pos, self.logvar[step].exp())

    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        flow_key = kwargs['flow_architecture']
        self.net = self.flow_dict[flow_key](n_blocks=n_blocks,
                                            input_size=input_size,
                                            hidden_size=hidden_size,
                                            n_hidden=n_hidden,
                                            cond_size=cond_size)

    def forward_separate(self, seq, cond):
        seq_prev, log_det_jacobian = self.net(seq, cond[:, None].expand(-1, seq.shape[1], -1))
        seq_log_det_jacobians = log_det_jacobian
        return seq_prev, seq_log_det_jacobians

    def inverse(self, pos, cond):
        seq = []
        seq_log_det_jacobians = []
        for step in range(self.pred_len):
            pos, log_det_jacobian = self.net.inverse(pos, cond)
            seq.append(pos)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq, dim=2)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=2)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians

    def log_prob(self, base_pos, seq, cond):
        return self.log_prob_separate(base_pos, seq, cond)

    def log_prob_separate(self, base_pos, seq, cond):
        seq_prev, seq_ldjs = self.forward_separate(seq, cond)
        base_pos = torch.cat([base_pos[:, None], seq[:, :-1]], dim=1)
        base_log_prob = self.base_dist(base_pos, step=list(range(base_pos.shape[1]))).log_prob(seq_prev)
        return base_log_prob.sum(dim=-1) + seq_ldjs

    def sample(self, base_pos, cond):
        base_pos_sample = self.base_dist(base_pos).sample()
        sample, _, _ = self.inverse(base_pos_sample, cond)
        return sample

    def sample_with_log_prob(self, base_pos, cond):
        base_pos_sample = self.base_dist(base_pos).sample()
        # grid_x, grid_y = torch.meshgrid(
        #     torch.linspace(-1, 1, 100),
        #     torch.linspace(-1, 1, 100))
        # grid_x = grid_x.to(self.logvar.device) * self.logvar[0].exp()[0] * 3
        # grid_y = grid_y.to(self.logvar.device) * self.logvar[0].exp()[1] * 3
        # grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        # base_pos_sample = base_pos + grid[None]
        
        sample, seq_ldjs_cumsum, seq_ldjs = self.inverse(base_pos_sample, cond)
        base_log_prob = self.base_dist(base_pos).log_prob(base_pos_sample)
        base_log_prob = base_log_prob.sum(dim=-1)[..., None]
        return sample, base_log_prob + seq_ldjs_cumsum, seq_ldjs, base_log_prob
        # return sample, base_log_prob.tile([1, 1, 12]), seq_ldjs, base_log_prob


class FlowChain_separate(FlowChain):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        flow_key = kwargs['flow_architecture']
        self.net = nn.ModuleList([self.flow_dict[flow_key](n_blocks=n_blocks,
                                                           input_size=input_size,
                                                           hidden_size=hidden_size,
                                                           n_hidden=n_hidden,
                                                           cond_size=cond_size)
                                  for _ in range(self.pred_len)])

    def forward_separate(self, seq, cond):
        seq_prev = []
        seq_log_det_jacobians = []
        for step in range(self.pred_len):
            pos_prev, log_det_jacobian = self.net[step](seq[:, step], cond)
            seq_prev.append(pos_prev)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_prev = torch.stack(seq_prev, dim=1)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians, dim=1)
        return seq_prev, seq_log_det_jacobians

    def inverse(self, pos, cond):
        seq = []
        seq_log_det_jacobians = []
        for step in range(self.pred_len):
            pos, log_det_jacobian = self.net[step].inverse(pos, cond)
            seq.append(pos)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq, dim=2)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=2)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=2)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians


class FlowChain_separate_cond(FlowChain_separate):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        flow_key = kwargs['flow_architecture']
        # increase cond_label_size for conditioning trajectory
        self.net = nn.ModuleList([self.flow_dict[flow_key](n_blocks=n_blocks,
                                                           input_size=input_size,
                                                           hidden_size=hidden_size,
                                                           n_hidden=n_hidden,
                                                           cond_size=cond_size + i * input_size)
                                  for i in range(self.pred_len)])
    
    def forward_separate(self, seq, cond):
        batch_size, pred_len, *_ = seq.shape
        seq_cond = seq.reshape(batch_size, pred_len * self.input_size)
        seq_prev = []
        seq_log_det_jacobians = []
        for step in range(self.pred_len):
            cond_cat = torch.cat(
                [cond, seq_cond[:, :step * self.input_size]], dim=-1)
            pos_prev, log_det_jacobian = self.net[step](seq[:, step], cond_cat)
            seq_prev.append(pos_prev)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_prev = torch.stack(seq_prev, dim=1)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians, dim=1)
        return seq_prev, seq_log_det_jacobians

    def inverse(self, pos, cond):
        seq = []
        seq_log_det_jacobians = []
        batch_size, n_sample, _ = pos.shape
        for step in range(self.pred_len):
            if not step == 0:
                seq_cond = torch.stack(seq, dim=2)
                seq_cond = seq_cond.reshape(batch_size, n_sample, -1)
                cond_cat = torch.cat([cond, seq_cond], dim=2)
                pos, log_det_jacobian = self.net[step].inverse(pos, cond_cat)
            else:
                pos, log_det_jacobian = self.net[step].inverse(
                    pos, cond)
            seq.append(pos)
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
                        cond_size=None,
                        batch_norm=True):

    modules = []
    mask = torch.arange(input_size).float() % 2
    for i in range(n_blocks):
        modules += [
            LinearMaskedCoupling(
                input_size, hidden_size, n_hidden, mask, cond_size
            )
        ]
        mask = 1 - mask
        modules += batch_norm * [BatchNorm(input_size)]

    return FlowSequential(*modules)


def create_MAF_step(n_blocks,
                    input_size,
                    hidden_size,
                    n_hidden,
                    cond_size=None,
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
                cond_size,
                activation,
                input_order,
                input_degrees,
            )
        ]
        input_degrees = modules[-1].input_degrees.flip(0)
        modules += batch_norm * [BatchNorm(input_size)]

    return FlowSequential(*modules)


class DiagonalGaussianConditionalDensity(nn.Module):
    def __init__(self,
                 cond_dim,
                 out_dim) -> None:
        super().__init__()
        self.shift_net = get_mlp(
            cond_dim, [out_dim] * 2, out_dim, nn.ReLU)

        self.log_scale_net = get_mlp(
            cond_dim, [out_dim] * 2, out_dim, nn.ReLU)

    def log_prob(self, u, cond_inputs):
        means, log_stddevs = self._means_and_log_stddevs(cond_inputs)
        return diagonal_gaussian_log_prob(u, means, log_stddevs)

    def sample(self, cond_inputs):
        means, log_stddevs = self._means_and_log_stddevs(cond_inputs)
        return diagonal_gaussian_sample(means, log_stddevs)

    def _means_and_log_stddevs(self, cond_inputs):
        return self.shift_net(cond_inputs), self.log_scale_net(cond_inputs)


def diagonal_gaussian_log_prob(u, means, log_stddevs):
    assert means.shape == log_stddevs.shape == u.shape

    vars = torch.exp(log_stddevs)**2

    *_, dim = u.shape

    const_term = -.5*dim*np.log(2*np.pi)
    log_det_terms = -torch.sum(log_stddevs, dim=-1)
    product_terms = -.5*torch.sum((u - means)**2 / vars, dim=-1)

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
                 cond_size) -> None:
        super().__init__()
        self.bijection = bijection
        self.cond_size = cond_size
        self.r_u_given_x = DiagonalGaussianConditionalDensity(
            input_size + cond_size, cond_size)
        self.q_u_given_z = DiagonalGaussianConditionalDensity(
            input_size + cond_size, cond_size)

    def forward(self, x, y):
        u, log_r_u = self.r_u_given_x.sample(torch.cat([x, y], dim=-1))
        z, log_det_jac = self.bijection.forward(x, u)
        log_q_u = self.q_u_given_z.log_prob(u, torch.cat([z, y], dim=-1))

        return z, log_det_jac + log_q_u - log_r_u

    def inverse(self, z, y):
        u, log_q_u = self.q_u_given_z.sample(torch.cat([z, y], dim=-1))
        x, log_det_jac = self.bijection.inverse(z, u)
        log_r_u = self.r_u_given_x.log_prob(u, torch.cat([x, y], dim=-1))
        
        return x, log_det_jac + log_q_u - log_r_u


class FlowChain_CIF(FlowChain):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        flow_key = kwargs['flow_architecture']
        bijection = self.flow_dict[flow_key](n_blocks=n_blocks,
                                             input_size=input_size,
                                             hidden_size=hidden_size,
                                             n_hidden=n_hidden,
                                             cond_size=cond_size)

        self.net = CIF_step(bijection,
                            input_size=input_size,
                            cond_size=cond_size)


class FlowChain_CIF_separate(FlowChain_separate):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        flow_key = kwargs['flow_architecture']
        self.net = nn.ModuleList([CIF_step(self.flow_dict[flow_key](n_blocks=n_blocks,
                                                                    input_size=input_size,
                                                                    hidden_size=hidden_size,
                                                                    n_hidden=n_hidden,
                                                                    cond_size=cond_size),
                                           input_size,
                                           cond_size)
                                  for _ in range(self.pred_len)])


class FlowChain_CIF_separate_cond(FlowChain_separate_cond):
    def build_net(self, n_blocks, input_size, hidden_size, n_hidden, cond_size, **kwargs):
        flow_key = kwargs['flow_architecture']
        self.net = nn.ModuleList([CIF_step(self.flow_dict[flow_key](n_blocks=n_blocks,
                                                                    input_size=input_size,
                                                                    hidden_size=hidden_size,
                                                                    n_hidden=n_hidden,
                                                                    cond_size=cond_size + i * input_size),
                                           input_size,
                                           cond_size + i * input_size)
                                  for i in range(self.pred_len)])
