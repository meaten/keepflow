from yacs.config import CfgNode
import math
from pathlib import Path
from typing import Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from utils import optimizer_to_cuda

from models.TP.TFCondARFlow import (
    FlowSequential, LinearMaskedCoupling, BatchNorm,
    MADE
)


class fastpredNF_TP(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(fastpredNF_TP, self).__init__()

        self.output_path = Path(cfg.OUTPUT_DIR)

        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len = cfg.DATA.PREDICT_LENGTH

        self.input_size = 2
        conditioning_length = 16
        self.d_model = 16
        num_heads = 4
        num_encoder_layers = 3
        num_decoder_layers = 3
        dim_feedforward_scale = 4
        dropout_rate = 0.1
        act_type = "gelu"
        target_dim = 2
        n_blocks = 3
        #n_blocks = 10
        n_hidden = 2
        #n_hidden = 6
        hidden_size = 64
        dequantize = False

        self.encoder_input = nn.Linear(self.input_size, self.d_model)
        self.decoder_input = nn.Linear(self.input_size, self.d_model)

        # [B, T, d_model] where d_model / num_heads is int
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward_scale * self.d_model,
            dropout=dropout_rate,
            activation=act_type,
        )
        
        model_dict = {
            "fastpredNF": fastpredNF,
            "fastpredNF_separate": fastpredNF_separate,
            "fastpredNF_separate_cond": fastpredNF_separate_cond,
            "fastpredNF_VFlow": fastpredNF_VFlow,
            "fastpredNF_VFlow_separate": fastpredNF_VFlow_separate,
            "fastpredNF_VFlow_separate_cond": fastpredNF_VFlow_separate_cond,
            "fastpredNF_CIF": fastpredNF_CIF,
            "fastpredNF_CIF_separate": fastpredNF_CIF_separate,
            "fastpredNF_CIF_separate_cond": fastpredNF_CIF_separate_cond
        }

        self.flow = model_dict[cfg.MODEL.TYPE](
            input_size=target_dim,
            n_blocks=n_blocks,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            cond_label_size=self.d_model,
        )
        self.dequantize = dequantize
        
        self.dist_args_proj = nn.Linear(self.d_model, conditioning_length)

        position = torch.arange(self.obs_len + self.pred_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2)
                             * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.obs_len + self.pred_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.optimizer = optim.Adam(self.parameters(), 5e-4)
        self.optimizers = [self.optimizer]
        
        self.__mean = nn.parameter.Parameter(torch.zeros(self.input_size), requires_grad=False)
        self.__std = nn.parameter.Parameter(torch.ones(self.input_size), requires_grad=False)
        
        self.ldjs_tmp = None
        
    @property
    def mean(self) -> torch.Tensor:
        return self.__mean
    
    @property
    def std(self) -> torch.Tensor:
        return self.__std
    
    @mean.setter
    def mean(self, mean: torch.Tensor) -> None:
        self.__mean = nn.parameter.Parameter(mean, requires_grad=False)
        
    @std.setter
    def std(self, std: torch.Tensor) -> None:
        self.__std = nn.parameter.Parameter(std, requires_grad=False)
        
    def norm_input(self, data_dict: Dict) -> Dict:
        for k in data_dict.keys():
            if k in ["obs", "gt"]:
                data_dict[k] = (data_dict[k] - self.mean) / self.std
            elif k[0] == "pred":
                data_dict[k] = (data_dict[k] - self.mean) / self.std
            elif k[0] == "prob":
                data_dict[k][..., :2] = (data_dict[k][..., :2] - self.mean) / self.std
                    
    def denorm_output(self, data_dict: Dict):
        for k in data_dict.keys():
            if k in ["obs", "gt"]:
                data_dict[k] = data_dict[k] * self.std + self.mean
            if k[0] == "pred":
                data_dict[k] = data_dict[k] * self.std + self.mean
            elif k[0] == "prob":
                data_dict[k][..., :2] = data_dict[k][..., :2] * self.std + self.mean

    def encode(self, data_dict: Dict) -> torch.Tensor:
        enc_inputs = data_dict["obs"]

        enc_pe = self.pe[: self.obs_len, ...]
        dec_pe = self.pe[-self.pred_len:, ...]
        
        enc_out = self.transformer.encoder(
            self.encoder_input(enc_inputs) + enc_pe
        )

        dec_output = self.transformer.decoder(
            dec_pe.expand(-1, enc_out.shape[1], -1),
            enc_out
        )

        return dec_output
    
    def predict(self, data_dict: Dict, return_prob: bool=True) -> Dict:
        if return_prob:
            return self.predict_inverse_prob(data_dict)
        else:
            return self.predict_inverse_ML(data_dict)
        #return self.predict_forward(data_dict)

    def predict_inverse_prob(self, data_dict: Dict) -> Dict:
        self.norm_input(data_dict)
        dec_output = self.encode(data_dict)

        sample_num = 10000
        
        pos = data_dict["obs"][-1][None].expand(sample_num, -1, -1).clone()
        dist_args = self.dist_args_proj(dec_output)[:, None].expand(-1, sample_num, -1, -1)
        sampled_seq, log_prob, seq_ldjs = self.flow.sample_with_log_prob(pos, cond=dist_args)
        
        self.ldjs_tmp = seq_ldjs
        
        data_dict[("prob", 0)] = torch.cat(
            [sampled_seq, torch.exp(log_prob)[..., None]], dim=-1)
        """
        index_ML = log_prob[-1].argmax(dim=0).unsqueeze(0).unsqueeze(-1)
        index_ML = index_ML.expand(data_dict["gt"].size()).unsqueeze(1)
        data_dict[("pred", 0)] = torch.gather(sampled_seq, 1, index_ML).squeeze(1)  # sample maximum likelihood
        """
        data_dict[("pred", 0)] = sampled_seq[:, -1]
        self.denorm_output(data_dict)
        return data_dict
    
    def predict_inverse_ML(self, data_dict: Dict) -> Dict:
        self.norm_input(data_dict)
        dec_output = self.encode(data_dict)

        sample_num = 100
        
        pos = data_dict["obs"][-1][None].expand(sample_num, -1, -1).clone()
        dist_args = self.dist_args_proj(dec_output)[:, None].expand(-1, sample_num, -1, -1)
        sampled_seq, log_prob, seq_ldjs = self.flow.sample_with_log_prob(pos, cond=dist_args)
        index_ML = log_prob[-1].argmax(dim=0).unsqueeze(0).unsqueeze(-1)
        index_ML = index_ML.expand(data_dict["obs"].size()).unsqueeze(1)
        data_dict[("pred", 0)] = torch.gather(sampled_seq, 1, index_ML).squeeze(1)  # sample maximum likelihood
        self.denorm_output(data_dict)
        return data_dict
    
    def predict_from_new_obs(self, data_dict: Dict, time_step: int) -> Dict:
        assert 0 < time_step and time_step < len(data_dict["obs"])
        self.norm_input(data_dict)
        data_dict[("prob", time_step)] = data_dict[("prob", 0)][time_step:].clone()
        
        base_log_prob = self.flow.base_dist(data_dict["gt"][time_step-1]).log_prob(data_dict[("prob", 0)][..., :2][time_step-1])
        base_log_prob = torch.sum(base_log_prob, dim=-1)
        log_prob = base_log_prob + torch.cumsum(self.ldjs_tmp[time_step:], dim=0) / torch.cumsum(torch.ones_like(self.ldjs_tmp[time_step:]), dim=0)
        data_dict[("prob", time_step)][..., -1] = torch.exp(log_prob)            
        
        index_ML = data_dict[("prob", time_step)][..., -1][-1].argmax(dim=0).unsqueeze(0).unsqueeze(-1)
        index_ML = index_ML.expand(data_dict["gt"][time_step:].size()).unsqueeze(1)
        data_dict[("pred", time_step)] = torch.gather(data_dict[("prob", time_step)][..., :2], 1, index_ML).squeeze(1)  # sample maximum likelihood
        
        self.denorm_output(data_dict)
        return data_dict
    
    def predict_forward(self, data_dict: Dict) -> Dict:
        # not for *_cond models
        data_dict = self.predict_inverse(data_dict)  # for data_dict["pred"]
        self.norm_input(data_dict)
        dec_output = self.encode(data_dict)
        
        sample_num_per_dim = 200
        batch_size = data_dict["obs"][-1].size()[0]
        pos = data_dict["obs"][-1][None].expand(sample_num_per_dim ** self.input_size, -1, -1).clone()
        dist_args = self.dist_args_proj(dec_output)[:, None].expand(-1, sample_num_per_dim ** self.input_size, -1, -1)
        seq = torch.cartesian_prod(torch.linspace(-3, 3, steps=sample_num_per_dim),
                                   torch.linspace(-3, 3, steps=sample_num_per_dim))[None, :, None].expand(self.pred_len, -1, batch_size, -1).cuda()
        
        log_prob = self.flow.log_prob_sequential(pos, seq, cond=dist_args)
        
        data_dict[("prob", 0)] = torch.cat([seq, torch.exp(log_prob)[..., None]], dim=-1)
        self.denorm_output(data_dict)
        return data_dict

    def update(self, data_dict: Dict) -> Dict:
        self.norm_input(data_dict)
        dec_output = self.encode(data_dict)

        dist_args = self.dist_args_proj(dec_output)

        gt = data_dict['gt']
        if self.dequantize:
            gt += torch.rand_like(data_dict['gt'])
        
        loss = -self.flow.log_prob(data_dict["obs"][-1], gt, dist_args)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.mean().item()}

    def save(self, epoch: int = 0, path: Path = None) -> None:
        if path is None:
            path = self.output_path / "ckpt.pt"

        ckpt = {
            'epoch': epoch,
            'state': self.state_dict(),
            'optim_state': self.optimizer.state_dict(),
        }

        torch.save(ckpt, path)
        
    def check_saved_path(self, path: Path = None) -> bool:
        if path is None:
            path = self.output_path / "ckpt.pt"        
        
        return path.exists()

    def load(self, path: Path = None) -> int:
        if path is None:
            path = self.output_path / "ckpt.pt"

        ckpt = torch.load(path)
        self.load_state_dict(ckpt['state'])

        self.optimizer.load_state_dict(ckpt['optim_state'])
        optimizer_to_cuda(self.optimizer)
        
        return ckpt["epoch"]

class fastpredNF(nn.Module):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int):
        super().__init__()

        self.__scale = None

        self.input_size = input_size
        self.net = create_RealNVP_step(n_blocks=n_blocks,
                                       input_size=self.input_size,
                                       hidden_size=hidden_size,
                                       n_hidden=n_hidden,
                                       cond_label_size=cond_label_size)
        # self.net = create_MAF_step(n_blocks=n_blocks,
        #                                input_size=self.input_size,
        #                                hidden_size=hidden_size,
        #                                n_hidden=n_hidden,
        #                                cond_label_size=cond_label_size)
        self.var = nn.parameter.Parameter(
            torch.ones(self.input_size)/10, requires_grad=True)
        
    def base_dist(self, pos):
        return Normal(pos, self.var)
        #return Normal(torch.zeros_like(pos), self.var)

    def forward_sequential(self, seq, cond):
        x = seq[-1:].detach().clone()
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            x, log_det_jacobian = self.net(x, cond[step:])
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x = torch.cat([seq[step-1][None], x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([torch.zeros(seq_log_det_jacobians_cumsum.shape[1:])[None],
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_u = x
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq_u, seq_log_det_jacobians_cumsum
    
    def forward_separate(self, seq, cond):
        seq_log_det_jacobians = 0
        seq_u, log_det_jacobian = self.net(seq, cond)
        seq_log_det_jacobians += log_det_jacobian
        return seq_u, torch.sum(seq_log_det_jacobians, dim=-1)

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        for step in range(cond.shape[0]):
            u, log_det_jacobian = self.net.inverse(u, cond[step])
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians = torch.sum(seq_log_det_jacobians, dim=-1)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians), dim=0)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians
    
    def log_prob(self, pos, x, cond):
        return self.log_prob_separate(pos, x, cond)
        #return self.log_prob_sequential(pos, x, cond)
    
    def log_prob_sequential(self, pos, x, cond):
        u, seq_ldjs_cumsum = self.forward_sequential(x, cond)
        base_log_prob = torch.sum(self.base_dist(pos).log_prob(u), dim=-1)
        return base_log_prob + seq_ldjs_cumsum

    def log_prob_separate(self, pos, x, cond):
        u, seq_ldjs = self.forward_separate(x, cond)
        pos = torch.cat([pos[None], x[:-1]], dim=0)
        base_log_prob = torch.sum(self.base_dist(pos).log_prob(u), dim=-1)
        return base_log_prob + seq_ldjs

    def sample(self, pos, cond):
        u = self.base_dist(pos).sample()
        sample, _ = self.inverse(u, cond)
        return sample

    def sample_with_log_prob(self, pos, cond):
        u = self.base_dist(pos).sample()
        sample, seq_ldjs_cumsum, seq_ldjs = self.inverse(u, cond)
        base_log_prob = torch.sum(self.base_dist(pos).log_prob(u), dim=-1) 
        return sample, base_log_prob + seq_ldjs_cumsum, seq_ldjs 
    
    
class fastpredNF_separate(fastpredNF):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 pred_len=8):
        super().__init__(input_size,
                         n_blocks,
                         hidden_size,
                         n_hidden,
                         cond_label_size)

        self.net = nn.ModuleList([create_RealNVP_step(n_blocks=n_blocks,
                                                      input_size=self.input_size,
                                                      hidden_size=hidden_size,
                                                      n_hidden=n_hidden,
                                                      cond_label_size=cond_label_size)
                                  for _ in range(pred_len)])
        # self.net = nn.ModuleList([create_MAF_step(n_blocks=n_blocks,
        #                                input_size=self.input_size,
        #                                hidden_size=hidden_size,
        #                                n_hidden=n_hidden,
        #                                cond_label_size=cond_label_size)
        #                           for _ in range(pred_len)])

    def forward_sequential(self, seq, cond):
        x = seq[-1:].detach().clone()
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            x, log_det_jacobian = self.net[step](x, cond[step:])
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x = torch.cat([seq[step-1][None], x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([torch.zeros(seq_log_det_jacobians_cumsum.shape[1:])[None],
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_u = x
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq_u, seq_log_det_jacobians_cumsum
    
    def forward_separate(self, seq, cond):
        seq_u = []
        seq_log_det_jacobians = []
        for step in range(seq.shape[0]):
            u, log_det_jacobian = self.net[step](seq[step], cond[step])
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq_u.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_u = torch.stack(seq_u)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        for step in range(cond.shape[0]):
            u, log_det_jacobian = self.net[step].inverse(u, cond[step])
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians
    
    
class fastpredNF_separate_cond(fastpredNF):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 pred_len=8):

        super().__init__(input_size,
                         n_blocks,
                         hidden_size,
                         n_hidden,
                         cond_label_size)
        
        # increase cond_label_size for conditioning trajectory
        self.net = nn.ModuleList([create_RealNVP_step(n_blocks=n_blocks,
                                                      input_size=self.input_size,
                                                      hidden_size=hidden_size,
                                                      n_hidden=n_hidden,
                                                      cond_label_size=cond_label_size + i * self.input_size)
                                  for i in range(pred_len)])

    def forward_sequential(self, seq, cond):
        x = seq[-1:].detach().clone()
        pred_len, batch_size, *_ = seq.shape
        seq_cond = seq.transpose(0, 1).reshape(batch_size, pred_len * self.input_size)[None]
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            cond_cat = torch.cat([cond[step:], 
                                  seq_cond[..., :step * self.input_size].expand(pred_len - step, -1, -1)],
                                 dim=-1)
            x, log_det_jacobian = self.net[step](x, cond_cat)
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x = torch.cat([seq[step-1][None], x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([torch.zeros(seq_log_det_jacobians_cumsum.shape[1:])[None],
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_u = x
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq_u, seq_log_det_jacobians_cumsum
    
    def forward_separate(self, seq, cond):
        pred_len, batch_size, *_ = seq.shape
        seq_cond = seq.transpose(0, 1).reshape(batch_size, pred_len * self.input_size)
        seq_u = []
        seq_log_det_jacobians = []
        for step in range(seq.shape[0]):
            cond_cat = torch.cat([cond[step], seq_cond[..., :step * self.input_size]], dim=-1)
            u, log_det_jacobian = self.net[step](seq[step], cond_cat)
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq_u.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_u = torch.stack(seq_u)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        n_sample, batch_size, _ = u.shape
        for step in range(cond.shape[0]):
            if not step == 0:
                seq_cond = torch.stack(seq).transpose(0,1).transpose(1,2)
                seq_cond = seq_cond.reshape(n_sample, batch_size, -1)
                cond_cat = torch.cat([cond[step], seq_cond], dim=-1)
                u, log_det_jacobian = self.net[step].inverse(u, cond_cat)
            else:
                u, log_det_jacobian = self.net[step].inverse(u, cond[step])
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
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
    
    
class fastpredNF_VFlow(fastpredNF):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 aug_size: int=4):
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
        self.register_buffer("sample_var", torch.ones(self.input_size) * 0.1)
        
    def sample_dist(self, pos):
        return Normal(pos, self.sample_var)
        
    def forward_sequential(self, seq, cond):
        x = seq[-1:].detach().clone()
        x, log_det_jacobian_aug = self.augment(x)
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            x, log_det_jacobian = self.net(x, cond[step:])
            seq_log_det_jacobians_cumsum += torch.sum(log_det_jacobian, dim=-1)
            if not step == 0:
                x_, log_det_jacobian = self.augment(seq[step-1][None])
                x = torch.cat([x_, x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([log_det_jacobian,
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq_u, log_det_jacobian = self.augment.split(x)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        return seq_u, seq_log_det_jacobians_cumsum
    
    def forward_separate(self, seq, cond):
        seq, seq_log_det_jacobians = self.augment(seq)
        seq_u, log_det_jacobian = self.net(seq, cond)
        seq_log_det_jacobians += torch.sum(log_det_jacobian, dim=-1)
        seq_u, log_det_jacobian = self.augment.split(seq_u)
        seq_log_det_jacobians += log_det_jacobian
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        u, log_det_jacobian_aug = self.augment(u)
        for step in range(cond.shape[0]):
            u, log_det_jacobian = self.net.inverse(u, cond[step])
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
                
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq, log_det_jacobian = self.augment.split(seq)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        # TODO: need to consider augmentation gap in 'seq_log_det_jacobians'
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians
    
    
class fastpredNF_VFlow_separate(fastpredNF_VFlow):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 aug_size: int=4,
                 pred_len=8):
        super().__init__(input_size,
                         n_blocks,
                         hidden_size,
                         n_hidden,
                         cond_label_size,
                         aug_size)
        
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
        x = seq[-1:].detach().clone()
        x, log_det_jacobian_aug = self.augment(x)
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            x, log_det_jacobian = self.net[step](x, cond[step:])
            seq_log_det_jacobians_cumsum += torch.sum(log_det_jacobian, dim=-1)
            if not step == 0:
                x_, log_det_jacobian = self.augment(seq[step-1][None])
                x = torch.cat([x_, x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([log_det_jacobian,
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq_u, log_det_jacobian = self.augment.split(x)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        return seq_u, seq_log_det_jacobians_cumsum
    
    def forward_separate(self, seq, cond):
        seq, seq_log_det_jacobians = self.augment(seq)
        
        seq_u = []
        seq_log_det_jacobians = []
        for step in range(seq.shape[0]):
            u, log_det_jacobian = self.net[step](seq[step], cond[step])
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq_u.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_u = torch.stack(seq_u)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        
        seq_u, log_det_jacobian = self.augment.split(seq_u)
        seq_log_det_jacobians += log_det_jacobian
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        u, log_det_jacobian_aug = self.augment(u)
        for step in range(cond.shape[0]):
            u, log_det_jacobian = self.net[step].inverse(u, cond[step])
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
                
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq, log_det_jacobian = self.augment.split(seq)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        # TODO: need to consider augmentation gap in 'seq_log_det_jacobians'
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians
    
    
class fastpredNF_VFlow_separate_cond(fastpredNF_VFlow):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 aug_size: int=4,
                 pred_len=8):
        super().__init__(input_size,
                         n_blocks,
                         hidden_size,
                         n_hidden,
                         cond_label_size,
                         aug_size)
        
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
        seq_cond = seq.transpose(0, 1).reshape(batch_size, pred_len * self.input_size)[None]
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            cond_cat = torch.cat([cond[step:], 
                                  seq_cond[..., :step * self.input_size].expand(pred_len - step, -1, -1)],
                                 dim=-1)
            x, log_det_jacobian = self.net[step](x, cond_cat)
            seq_log_det_jacobians_cumsum += torch.sum(log_det_jacobian, dim=-1)
            if not step == 0:
                x_, log_det_jacobian = self.augment(seq[step-1][None])
                x = torch.cat([x_, x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([log_det_jacobian,
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        seq_log_det_jacobians_cumsum += log_det_jacobian_aug
        seq_u, log_det_jacobian = self.augment.split(x)
        seq_log_det_jacobians_cumsum += log_det_jacobian
        
        return seq_u, seq_log_det_jacobians_cumsum
    
    def forward_separate(self, seq, cond):
        pred_len, batch_size, *_ = seq.shape
        seq_cond = seq.transpose(0, 1).reshape(batch_size, pred_len * self.input_size)
        seq, log_det_jacobians_aug = self.augment(seq)
        seq_u = []
        seq_log_det_jacobians = []
        for step in range(seq.shape[0]):
            cond_cat = torch.cat([cond[step], seq_cond[..., :step * self.input_size]], dim=-1)
            u, log_det_jacobian = self.net[step](seq[step], cond_cat)
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
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
                seq_cond = torch.stack(seq_c).transpose(0,1).transpose(1,2)
                seq_cond = seq_cond.reshape(n_sample, batch_size, -1)
                cond_cat = torch.cat([cond[step], seq_cond], dim=-1)
                u, log_det_jacobian = self.net[step].inverse(u, cond_cat)
            else:
                u, log_det_jacobian = self.net[step].inverse(u, cond[step])
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
            seq.append(u)
            seq_c.append(u[..., :self.input_size])
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
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
        log_softmax_outputs=False
):
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
        self.r_u_given_z = DiagonalGaussianConditionalDensity(input_size + cond_label_size, cond_label_size)
        self.q_u_given_w = DiagonalGaussianConditionalDensity(input_size + cond_label_size, cond_label_size)
        
    def forward(self, z, y):
        u, log_r_u = self.r_u_given_z.sample(torch.cat([z, y], dim=-1))
        w, log_det_jac = self.bijection.forward(z, u)
        log_q_u = self.q_u_given_w.log_prob(u, torch.cat([w, y], dim=-1))
        
        return w, torch.sum(log_det_jac, dim=-1) + log_q_u - log_r_u
    
    def inverse(self, w, y):
        u, log_q_u = self.q_u_given_w.sample(torch.cat([w, y], dim=-1))
        z, log_det_jac = self.bijection.inverse(w, u)
        log_r_u = self.r_u_given_z.log_prob(u, torch.cat([z, y], dim=-1))

        return z, torch.sum(log_det_jac, dim=-1) + log_q_u - log_r_u
    
    
class BatchNorm_reduce(BatchNorm):
    def forward(self, x, cond_y=None):
        y, log_abs_det_jacobian = super().forward(x, cond_y)
        return y, torch.sum(log_abs_det_jacobian, dim=-1)
    
    def inverse(self, y, cond_y=None):
        x, log_abs_det_jacobian = super().inverse(y, cond_y)
        return x, torch.sum(log_abs_det_jacobian, dim=-1)
    
def create_RealNVP_CIF_step(n_blocks,
                        input_size,
                        hidden_size,
                        n_hidden,
                        cond_label_size=None,
                        batch_norm=True):

    modules = []
    mask = torch.arange(input_size).float() % 2
    for i in range(n_blocks):
        modules += [
            CIF_step(LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size),
                     input_size,
                     cond_label_size)
        ]
        mask = 1 - mask
        modules += batch_norm * [BatchNorm_reduce(input_size)]

    return FlowSequential(*modules)


class fastpredNF_CIF(fastpredNF):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int):
        super().__init__(
            input_size,
            n_blocks,
            hidden_size,
            n_hidden,
            cond_label_size
        )
        """
        bijection = create_RealNVP_step(n_blocks=n_blocks,
                                       input_size=self.input_size,
                                       hidden_size=hidden_size,
                                       n_hidden=n_hidden,
                                       cond_label_size=cond_label_size)
        
        self.net = CIF_step(bijection,
                            input_size=self.input_size,
                            cond_label_size=cond_label_size)
        """
        
        self.net = create_RealNVP_CIF_step(n_blocks=n_blocks,
                                       input_size=self.input_size,
                                       hidden_size=hidden_size,
                                       n_hidden=n_hidden,
                                       cond_label_size=cond_label_size)
        
    def forward_sequential(self, seq, cond):
        x = seq[-1:].detach().clone()
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            x, log_det_jacobian = self.net(x, cond[step:])
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x = torch.cat([seq[step-1][None], x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([torch.zeros(seq_log_det_jacobians_cumsum.shape[1:])[None],
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_u = x
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq_u, seq_log_det_jacobians_cumsum
    
    def forward_separate(self, seq, cond):
        seq_log_det_jacobians = 0
        seq_u, log_det_jacobian = self.net(seq, cond)
        seq_log_det_jacobians += log_det_jacobian
        return seq_u, seq_log_det_jacobians
        
    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        for step in range(cond.shape[0]):
            u, log_det_jacobian = self.net.inverse(u, cond[step])
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians
    
    
class fastpredNF_CIF_separate(fastpredNF_separate):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 pred_len=8):
        super().__init__(
            input_size,
            n_blocks,
            hidden_size,
            n_hidden,
            cond_label_size,
            pred_len
        )
        
        self.net = nn.ModuleList([CIF_step(create_RealNVP_step(n_blocks=n_blocks,
                                                      input_size=self.input_size,
                                                      hidden_size=hidden_size,
                                                      n_hidden=n_hidden,
                                                      cond_label_size=cond_label_size),
                                           input_size,
                                           cond_label_size)
                                  for _ in range(pred_len)])
    
    def forward_sequential(self, seq, cond):
        x = seq[-1:].detach().clone()
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            x, log_det_jacobian = self.net[step](x, cond[step:])
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x = torch.cat([seq[step-1][None], x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([torch.zeros(seq_log_det_jacobians_cumsum.shape[1:])[None],
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_u = x
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq_u, seq_log_det_jacobians_cumsum
    
    def forward_separate(self, seq, cond):
        seq_u = []
        seq_log_det_jacobians = []
        for step in range(seq.shape[0]):
            u, log_det_jacobian = self.net[step](seq[step], cond[step])
            seq_u.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_u = torch.stack(seq_u)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        for step in range(cond.shape[0]):
            u, log_det_jacobian = self.net[step].inverse(u, cond[step])
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians
    
    
class fastpredNF_CIF_separate_cond(fastpredNF):
    def __init__(self,
                 input_size: int,
                 n_blocks: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int,
                 pred_len=8):
        super().__init__(
            input_size,
            n_blocks,
            hidden_size,
            n_hidden,
            cond_label_size
        )
        
        self.net = nn.ModuleList([CIF_step(create_RealNVP_step(n_blocks=n_blocks,
                                                      input_size=self.input_size,
                                                      hidden_size=hidden_size,
                                                      n_hidden=n_hidden,
                                                      cond_label_size=cond_label_size + i * self.input_size),
                                           input_size,
                                           cond_label_size + i * self.input_size)
                                  for i in range(pred_len)])
        
    def forward_sequential(self, seq, cond):
        x = seq[-1:].detach().clone()
        pred_len, batch_size, *_ = seq.shape
        seq_cond = seq.transpose(0, 1).reshape(batch_size, pred_len * self.input_size)[None]
        seq_log_det_jacobians_cumsum = 0
        for step in range(seq.shape[0]-1, -1, -1):
            cond_cat = torch.cat([cond[step:], 
                                  seq_cond[..., :step * self.input_size].expand(pred_len - step, -1, -1)],
                                 dim=-1)
            x, log_det_jacobian = self.net[step](x, cond_cat)
            seq_log_det_jacobians_cumsum += log_det_jacobian
            if not step == 0:
                x = torch.cat([seq[step-1][None], x], dim=0)
                seq_log_det_jacobians_cumsum = torch.cat([torch.zeros(seq_log_det_jacobians_cumsum.shape[1:])[None],
                                                   seq_log_det_jacobians_cumsum], dim=0)
        seq_u = x
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq_u, seq_log_det_jacobians_cumsum
    
    def forward_separate(self, seq, cond):
        pred_len, batch_size, *_ = seq.shape
        seq_cond = seq.transpose(0, 1).reshape(batch_size, pred_len * self.input_size)
        seq_u = []
        seq_log_det_jacobians = []
        for step in range(seq.shape[0]):
            cond_cat = torch.cat([cond[step], seq_cond[..., :step * self.input_size]], dim=-1)
            u, log_det_jacobian = self.net[step](seq[step], cond_cat)
            seq_u.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq_u = torch.stack(seq_u)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        return seq_u, seq_log_det_jacobians

    def inverse(self, u, cond):
        seq = []
        seq_log_det_jacobians = []
        n_sample, batch_size, _ = u.shape
        for step in range(cond.shape[0]):
            if not step == 0:
                seq_cond = torch.stack(seq).transpose(0,1).transpose(1,2)
                seq_cond = seq_cond.reshape(n_sample, batch_size, -1)
                cond_cat = torch.cat([cond[step], seq_cond], dim=-1)
                u, log_det_jacobian = self.net[step].inverse(u, cond_cat)
            else:
                u, log_det_jacobian = self.net[step].inverse(u, cond[step])
            seq.append(u)
            seq_log_det_jacobians.append(log_det_jacobian)
        seq = torch.stack(seq)
        seq_log_det_jacobians = torch.stack(seq_log_det_jacobians)
        seq_log_det_jacobians_cumsum = torch.cumsum(
            seq_log_det_jacobians, dim=0)
        seq_log_det_jacobians_cumsum /= torch.cumsum(torch.ones_like(seq_log_det_jacobians_cumsum), dim=0)
        return seq, seq_log_det_jacobians_cumsum, seq_log_det_jacobians