import sys
from yacs.config import CfgNode
from typing import Dict, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from keepflow.models import ModelTemplate
sys.path.append('extern/traj/sgan')
from sgan.models import TrajectoryGenerator, SocialPooling
from sgan.losses import l2_loss
from scripts.train import init_weights


class socialLSTM(ModelTemplate):
    def __init__(self, cfg: CfgNode) -> None:
        super(socialLSTM, self).__init__(cfg)

        self.output_path = Path(cfg.SAVE_DIR)

        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len = cfg.DATA.PREDICT_LENGTH
        
        self.g = CustomTrajectoryGenerator(
            obs_len=cfg.DATA.OBSERVE_LENGTH,
            pred_len=cfg.DATA.PREDICT_LENGTH,
            embedding_dim=16,
            encoder_h_dim=32,
            decoder_h_dim=32,
            mlp_dim=64,
            num_layers=1,
            noise_dim=(0,),
            noise_type='gaussian',
            noise_mix_type='global',
            pooling_type='spool',
            pool_every_timestep=False,
            dropout=0.0,
            bottleneck_dim=32,
            neighborhood_size=3.0,
            grid_size=8,
            batch_norm=False,
            device=cfg.DEVICE
        )

        self.g.apply(init_weights).type(torch.FloatTensor).train()

        self.optimizer_g = optim.Adam(self.g.parameters(), 5e-4)

        self.optimizers = [self.optimizer_g]

        self.clipping_threshold_g = 1.5
        self.l2_loss_weight = 1.0

        self.pos_idx = [0, 1]
        self.vel_idx = [2, 3]

    def predict(self, data_dict: Dict, return_prob=False) -> Dict:
        obs_slstm, obs_slstm_rel, seq_start_end_slstm = self.data_dict_to_slstm(data_dict)
        pred_traj_fake_rel = self.g.forward(obs_slstm.permute(1, 0, 2).contiguous(),
                                        obs_slstm_rel.permute(1, 0, 2).contiguous(),
                                        seq_start_end_slstm)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1, 0, 2)
        data_dict[("pred", 0)] = pred_traj_fake_rel[seq_start_end_slstm[:, 0]]
        return data_dict
    
    def predict_from_new_obs(self, data_dict: Dict, time_step: int) -> Dict:
        # do nothing
        return data_dict

    def data_dict_to_slstm(self, data_dict):
        bs, t, _ = data_dict["obs"].shape
        obs = data_dict['obs']
        neighbors = data_dict["neighbors"]
        neighbor_type = data_dict['neighbor_type']

        seq_end = torch.arange(bs) + torch.cumsum(torch.LongTensor([len(n[nt != -1]) for n, nt in zip(neighbors, neighbor_type)]), dim=0) + 1
        seq_start = torch.concat([torch.LongTensor([0]), seq_end[:-1]])
        seq_start_end_slstm = torch.cat([seq_start[:, None], seq_end[:, None]], dim=1)

        list_o_n = []
        for o, n, nt in zip(obs, neighbors, neighbor_type):
            list_o_n.append(o[None])
            list_o_n.append(n[nt != -1])
        obs_slstm = torch.concat(list_o_n, dim=0)
        # assert torch.all(obs_slstm[seq_start_end_slstm[:, 0]] == data_dict["obs"])
        obs_slstm_rel = obs_slstm[..., self.vel_idx]
        obs_slstm = obs_slstm[..., self.pos_idx]
        
        return obs_slstm, obs_slstm_rel, seq_start_end_slstm

    def update(self, data_dict: Dict) -> Dict:
        g_loss = 0
        
        data_dict = self.predict(data_dict)
        loss_mask = ~torch.isnan(data_dict['gt'])
        if self.l2_loss_weight > 0:
            g_l2_loss_rel = self.l2_loss_weight * (data_dict[('pred', 0)] - data_dict['gt'].nan_to_num()) ** 2
            g_loss = g_l2_loss_rel
            g_loss *= loss_mask
            g_loss = g_loss.mean()
        self.optimizer_g.zero_grad()
        g_loss.backward()
        if self.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                self.g.parameters(), self.clipping_threshold_g
            )
        self.optimizer_g.step()
        
        return {"g_loss": g_loss.item()}

    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.output_path / "ckpt.pt"
            
        ckpt = {
            'epoch': epoch,
            'g_state': self.g.state_dict(),
            'g_optim_state': self.optimizer_g.state_dict(),
        }

        torch.save(ckpt, path)

    def load(self, path: Path=None) -> int:
        if path is None:
            path = self.output_path / "ckpt.pt"
        
        ckpt = torch.load(path)
        self.g.load_state_dict(ckpt['g_state'])

        self.optimizer_g.load_state_dict(ckpt['g_optim_state'])
        
        return ckpt["epoch"]
    

class CustomTrajectoryGenerator(TrajectoryGenerator):
    def __init__(self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,
        device='cuda:0'):
        super().__init__( obs_len, pred_len, embedding_dim, encoder_h_dim,
        decoder_h_dim, mlp_dim, num_layers, noise_dim,
        noise_type, noise_mix_type, pooling_type,
        pool_every_timestep, dropout, bottleneck_dim,
        activation, batch_norm, neighborhood_size, grid_size)

        if pooling_type == 'spool':
            self.pool_net = CustomSocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size,
                device=device
            )

class CustomSocialPooling(SocialPooling):
    def __init__(
        self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None, device='cuda:0'
    ):
        super().__init__(
            h_dim, activation, batch_norm, dropout,
            neighborhood_size, grid_size, pool_dim
        )

        self.device = device

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)
            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos.to(self.device),
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h

