import sys
from yacs.config import CfgNode
from typing import Dict
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from .social_lstm import socialLSTM, CustomTrajectoryGenerator

sys.path.append('extern/traj/sgan')
from sgan.models import TrajectoryDiscriminator
from sgan.losses import gan_g_loss, gan_d_loss
from scripts.train import init_weights, generator_step, discriminator_step


class socialGAN(socialLSTM):
    def __init__(self, cfg: CfgNode) -> None:
        super(socialGAN, self).__init__(cfg)

        self.g_step = 1

        self.g = CustomTrajectoryGenerator(
            obs_len=cfg.DATA.OBSERVE_LENGTH,
            pred_len=cfg.DATA.PREDICT_LENGTH,
            embedding_dim=16,
            encoder_h_dim=32,
            decoder_h_dim=32,
            mlp_dim=64,
            num_layers=1,
            noise_dim=(8,),
            noise_type='gaussian',
            noise_mix_type='global',
            pooling_type='none',
            pool_every_timestep=False,
            dropout=0.0,
            bottleneck_dim=0,
            neighborhood_size=2.0,
            grid_size=8,
            batch_norm=False,
            device=cfg.DEVICE
        )

        self.g.apply(init_weights).type(torch.FloatTensor).train()

        self.d_step = 2

        self.d = TrajectoryDiscriminator(
            obs_len=cfg.DATA.OBSERVE_LENGTH,
            pred_len=cfg.DATA.PREDICT_LENGTH,
            embedding_dim=16,
            h_dim=64,
            mlp_dim=64,
            num_layers=1,
            batch_norm=False,
            dropout=0.0,
            d_type='global'
        )

        self.d.apply(init_weights).type(torch.FloatTensor).train()

        self.optimizer_g = optim.Adam(self.g.parameters(), lr=5e-4)
        self.optimizer_d = optim.Adam(self.d.parameters(), lr=5e-4)

        self.optimizers = [self.optimizer_g, self.optimizer_d]

        self.best_k = 20
        self.clipping_threshold_d = 0.0

    def training_step(self, data_dict: Dict) -> Dict:
        obs_slstm, obs_slstm_rel, gt_slstm, gt_slstm_rel, loss_mask, seq_start_end_slstm = self.data_dict_to_slstm(data_dict)
        batch = (obs_slstm, gt_slstm, obs_slstm_rel, gt_slstm_rel, torch.Tensor([]), loss_mask, seq_start_end_slstm)
        
        # d_step_left = self.d_step
        # g_step_left = self.g_step
        # while True:
        #     if d_step_left > 0:
        #         losses_d = discriminator_step(
        #             self, batch, self.g,
        #             self.d, gan_d_loss,
        #             self.optimizer_d)
        #         d_step_left -= 1

        #     if g_step_left > 0:    
        #         losses_g = generator_step(
        #             self, batch, self.g,
        #             self.d, gan_g_loss,
        #             self.optimizer_g)
        #         g_step_left -= 1

        #     if d_step_left == 0 and g_step_left == 0:
        #         break

        for _ in range(self.d_step):
            losses_d = discriminator_step(
            self, batch, self.g,
            self.d, gan_d_loss,
            self.optimizer_d)

        for _ in range(self.g_step):
            losses_g = generator_step(
                self, batch, self.g,
                self.d, gan_g_loss,
                self.optimizer_g)

        return {"loss": losses_g['G_total_loss'] + losses_d['D_total_loss']}