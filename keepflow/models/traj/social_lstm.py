import sys
from yacs.config import CfgNode
from typing import Dict
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from keepflow.models import ModelTemplate
sys.path.append('extern/traj/sgan')
from sgan.models import TrajectoryGenerator, SocialPooling
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


    def predict(self, data_dict: Dict, return_prob=False) -> Dict:
        obs_slstm, obs_slstm_rel, gt_slstm, gt_slstm_rel, loss_mask, seq_start_end_slstm = self.data_dict_to_slstm(data_dict)
        
        n_sample = self.kde_n_sample if return_prob else 20

        # pred = []
        # for _ in range(n_sample):
        #     pred_traj_fake_rel = self.g.forward(obs_slstm,
        #                                     obs_slstm_rel,
        #                                     seq_start_end_slstm)
        #     pred_traj_fake_rel = pred_traj_fake_rel.permute(1, 0, 2)[seq_start_end_slstm[:, 0]]
        #     pred_traj_fake = relative_to_abs(pred_traj_fake_rel, data_dict['obs'][:, -1])
        #     pred.append(pred_traj_fake)
        # pred = torch.stack(pred, dim=1)
        # data_dict[("pred", 0)] = pred[:, :20]
        pred_traj_fake_rel = self.g.forward(
            obs_slstm,
            obs_slstm_rel,
            seq_start_end_slstm,
            n_sample=n_sample
        )

        max_val = torch.max(seq_start_end_slstm)
        tiled_start_end = seq_start_end_slstm.repeat(n_sample, 1)
        offsets = torch.arange(n_sample).repeat_interleave(len(seq_start_end_slstm)) * max_val
        seq_start_end = tiled_start_end + offsets[:, None]

        pred_traj_fake_rel = pred_traj_fake_rel.permute(1, 0, 2)[seq_start_end[:, 0]]
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, data_dict['obs'][:, -1].repeat(n_sample, 1))
        pred = pred_traj_fake.reshape([-1, n_sample, self.pred_len, 2])
        data_dict[("pred", 0)] = pred[:, :20]

        if return_prob:
            traj_pred = pred
            traj_pred = torch.cat([traj_pred, torch.zeros_like(traj_pred)], dim=3)
            data_dict[("prob", 0)] = traj_pred[..., :3]
            
        return data_dict

    def data_dict_to_slstm(self, data_dict):
        bs, t, _ = data_dict["obs"].shape
        obs = data_dict['obs']
        neighbors = data_dict["neighbors"]
        neighbors_gt = data_dict["neighbors_gt"]
        neighbor_type = data_dict['neighbor_type']

        seq_end = torch.arange(bs) + torch.cumsum(torch.LongTensor([len(n[nt != -1]) for n, nt in zip(neighbors, neighbor_type)]), dim=0) + 1
        seq_start = torch.concat([torch.LongTensor([0]), seq_end[:-1]])
        seq_start_end_slstm = torch.cat([seq_start[:, None], seq_end[:, None]], dim=1)

        list_o_n = []
        for o, n, nt in zip(obs, neighbors, neighbor_type):
            list_o_n.append(o[None])
            list_o_n.append(n[nt != -1])
        
        obs_slstm = torch.concat(list_o_n, dim=0).nan_to_num()
        obs_slstm_rel = torch.zeros_like(obs_slstm).cuda()
        obs_slstm_rel[:, 1:] = obs_slstm[:, 1:] - obs_slstm[:, :-1]

        list_o_n = []
        for o, n, nt in zip(data_dict["gt"], neighbors_gt, neighbor_type):
            list_o_n.append(o[None])
            list_o_n.append(n[nt != -1])
        
        gt_slstm = torch.concat(list_o_n, dim=0).nan_to_num()
        gt_slstm_rel = gt_slstm - torch.cat([obs_slstm[:, -1][:, None], gt_slstm[:, :-1]], dim=1)

        assert torch.all(obs_slstm[seq_start_end_slstm[:, 0]] == data_dict["obs"])
        assert torch.all(gt_slstm[seq_start_end_slstm[:, 0]] == data_dict["gt"])

        loss_mask = ~torch.isnan(torch.cat([obs_slstm, gt_slstm], dim=1)[..., 0])
        
        return (
            obs_slstm.permute(1, 0, 2).contiguous(),
            obs_slstm_rel.permute(1, 0, 2).contiguous(),
            gt_slstm.permute(1, 0, 2).contiguous(),
            gt_slstm_rel.permute(1, 0, 2).contiguous(),
            loss_mask,
            seq_start_end_slstm
        )
    
    
    def training_step(self, data_dict: Dict) -> Dict:
        g_loss = 0
        
        data_dict = self.predict(data_dict, n_sample=1)
        loss_mask = ~torch.isnan(data_dict['gt'])
        if self.l2_loss_weight > 0:
            g_l2_loss_rel = self.l2_loss_weight * (data_dict[('pred', 0)].squeeze() - data_dict['gt'].nan_to_num()) ** 2
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
        
        return {"loss": g_loss.item()}

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
    
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None, n_sample=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
        
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        
        if n_sample is not None:
            noise_input = noise_input.tile(n_sample, 1)
            # seq_start_end = seq_start_end.tile(n_sample, 1).flatten().cumsum(dim=0).reshape(-1,2)
            max_val = torch.max(seq_start_end)
            tiled_start_end = seq_start_end.repeat(n_sample, 1)
            offsets = torch.arange(n_sample).repeat_interleave(len(seq_start_end)) * max_val
            seq_start_end = tiled_start_end + offsets[:, None]
            batch *= n_sample
            obs_traj = obs_traj.tile(1, n_sample, 1)
            obs_traj_rel = obs_traj_rel.tile(1, n_sample, 1)            
        
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel
    
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


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj