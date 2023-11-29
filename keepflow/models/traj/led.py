import sys
from pathlib import Path
import torch
import numpy as np

from keepflow.models import ModelTemplate
sys.path.append('extern/traj/LED')
from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
from trainer.train_led_trajectory_augment_input import Trainer

sys.path.append('extern/traj/MID')
from dataset import restore

class LED(ModelTemplate, Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.n_steps = 100

        self.betas = self.make_beta_schedule(
			schedule='linear', n_timesteps=self.n_steps, 
			start=1.e-4, end=5.e-2).to(self.device)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        self.model = CoreDenoisingModel().to(self.device)
        model_cp = torch.load('extern/traj/LED/results/checkpoints/base_diffusion_model.p', map_location='cpu')
        self.model.load_state_dict(model_cp['model_dict'])
  
        self.model_initializer = InitializationModel(t_h=10, d_h=6, t_f=20, d_f=2, k_pred=20).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model_initializer.parameters(),                                            
                                           lr=1.e-3)
        self.optimizers = [self.optimizer]
        
        self.temporal_reweight = torch.FloatTensor([i for i in reversed(range(1, self.pred_len + 1))]).to(self.device).unsqueeze(0).unsqueeze(0) / 10

    def data_preprocess(self, data_dict):
        obs_st = data_dict['obs_st'].clone().cpu()
        neighbors_st = restore(data_dict['neighbors_st'])[('PEDESTRIAN', 'PEDESTRIAN')]
        
        batch_size = data_dict['obs'].shape[0] + sum([len(n) for n in neighbors_st])
        
        partitioning_index = np.cumsum([1] + [len(n) + 1 for n in neighbors_st[:-1]]) - 1
        traj_mask = torch.zeros(batch_size, batch_size).to(self.device)
        for i in range(len(partitioning_index) - 1):
            traj_mask[partitioning_index[i]:partitioning_index[i+1], partitioning_index[i]:partitioning_index[i+1]] = 1.
        traj_mask[partitioning_index[-1]:batch_size, partitioning_index[-1]:batch_size] = 1
        
        past_traj = torch.cat([torch.stack([o] + n) for o, n in zip(obs_st, neighbors_st)]).to(self.device)
        
        past_traj = torch.nn.functional.pad(past_traj, (0, 0, 10 - self.obs_len, 0), 'replicate')
        fut_traj = data_dict['gt_st']
        return partitioning_index, batch_size, traj_mask, past_traj, fut_traj

    def predict(self, data_dict, return_prob=False):
        partitioning_index, batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data_dict)

        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)    
        sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        loc = sample_prediction + mean_estimation[:, None]
    
        pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
        pred_traj = pred_traj[partitioning_index][:, :, :self.pred_len]
        
        idx = np.random.randint(pred_traj.shape[1])
        data_dict[('pred_st', 0)] = pred_traj[:, idx]
        
        if return_prob:
            pred_traj = torch.cat([pred_traj, torch.zeros_like(pred_traj)], dim=3)
            data_dict[("prob_st", 0)] = pred_traj[..., :3]
        
        return data_dict

    def update(self, data_dict) -> None:
        partitioning_index, batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data_dict)
        
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
        sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        loc = sample_prediction + mean_estimation[:, None]
        
        generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
        generated_y = generated_y[partitioning_index][:, :, :self.pred_len]
        
        loss_dist = ((generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1) * self.temporal_reweight
                    ).mean(dim=-1).min(dim=1)[0].mean()
        loss_uncertainty = (torch.exp(-variance_estimation) * (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2)) 
                                + 
                            variance_estimation).mean()
        
        loss = loss_dist*50 + loss_uncertainty
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.model_path
            
        ckpt = {
            'epoch': epoch,
            'model': self.model_initializer.state_dict(),
            'optim_state': self.optimizer.state_dict(),
        }

        torch.save(ckpt, path)
    

    def load(self, path: Path=None) -> int:
        if path is None:
            path = self.model_path
        
        ckpt = torch.load(path)
        self.model_initializer.load_state_dict(ckpt['model'])

        self.optimizer.load_state_dict(ckpt['optim_state'])
    
        return ckpt['epoch']
    
    def check_saved_path(self, path: Path = None) -> bool:
        if path is None:
            path = self.model_path

        return path.exists()