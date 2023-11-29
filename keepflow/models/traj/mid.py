import sys
import yaml
from pathlib import Path
from easydict import EasyDict
import torch
import torch.optim as optim

from keepflow.models import ModelTemplate
sys.path.append('extern/traj/MID')
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
from models.trajectron import Trajectron
from models.encoders import MultimodalGenerativeCVAE
from models.autoencoder import AutoEncoder
from dataset import restore


class MID(ModelTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        with open("extern/traj/MID/configs/baseline.yaml") as f:
            config = yaml.safe_load(f)
        self.config = EasyDict(config)
        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # self.hyperparams['state'] = self.hyperparams[cfg.DATA.TP.STATE]
        # self.hyperparams['pred_state'] = self.hyperparams[cfg.DATA.TP.PRED_STATE]
            
        self.registrar = ModelRegistrar(self.save_dir, self.device)
        
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        import dill
        env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "processed_data" / f"{cfg.DATA.DATASET_NAME}_train.pkl"
        with open(env_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        self.encoder.set_environment(train_env)
        self.encoder.set_annealing_params()
        
        self.model = CustomAutoEncoder(self.config, encoder=self.encoder)
        
        self.pred_state = self.hyperparams['pred_state']
        self.optimizer = dict()
        for node_type in train_env.NodeType:
            if node_type not in self.pred_state:
                continue
            self.optimizer[node_type] = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                                    # {'params': self.registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008},
                                                    {'params': self.model.parameters()}
                                                    ],
                                                   lr=self.config.lr)
        
        self.optimizers = self.optimizer.values()
        
    def to(self, device):
        self.model.to(device)
        self.registrar.to(device)
        
    def eval(self):
        self.model.eval()
        self.registrar.eval()
        
        
    def train(self):
        self.model.train()
        self.registrar.train()
    
    def predict(self, data_dict, return_prob=False):
        batch = (data_dict["first_history_index"],
                 data_dict["obs"],
                 data_dict["gt"],
                 data_dict["obs_st"],
                 data_dict["gt_st"],
                 restore(data_dict["neighbors_st"]),
                 restore(data_dict["neighbors_edge"]),
                 data_dict["robot_traj_st"],
                 data_dict["map"])
        
        n_sample = 10000 if return_prob else 1
        
        node_type = "PEDESTRIAN"
        traj_pred = self.model.generate(batch, node_type, num_points=12, sample=n_sample, bestof=True)
        traj_pred = traj_pred.permute(1, 0, 2, 3)
        
        data_dict[("pred_st", 0)] = traj_pred[:, 0]
        
        if return_prob:
            traj_pred = torch.cat([traj_pred, torch.zeros_like(traj_pred)], dim=3)
            data_dict[("prob_st", 0)] = traj_pred[..., :3]
        
        return data_dict
    
    def predict_from_new_obs(self, data_dict, time_step: int):
        # TODO: need to implement the density estimation & update
        """
        from data.TP.preprocessing import data_dict_to_next_step
        data_dict_ = data_dict_to_next_step(data_dict, time_step)
        data_dict_ = self.predict(data_dict_, return_prob=True)
        data_dict[("pred_st", time_step)] = data_dict_[("pred_st", 0)][:, :-time_step]
        data_dict[("prob_st", time_step)] = data_dict_[("prob_st", 0)][:, :-time_step]
        """
        return data_dict
    
    def update(self, data_dict):
        batch = (data_dict["first_history_index"],
                 data_dict["obs"],
                 data_dict["gt"],
                 data_dict["obs_st"],
                 data_dict["gt_st"],
                 restore(data_dict["neighbors_st"]),
                 restore(data_dict["neighbors_edge"]),
                 data_dict["robot_traj_st"],
                 data_dict["map"])
        
        node_type = "PEDESTRIAN"
        self.optimizer[node_type].zero_grad()
        train_loss = self.model.get_loss(batch, node_type)
        train_loss.backward()
        self.optimizer[node_type].step()
        
        return {"loss": train_loss.item()}
    
    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.model_path

        ckpt = {
            'epoch': epoch,
            'encoder': self.registrar.model_dict,
            'model': self.model.state_dict(),
            'optim_state': self.optimizer["PEDESTRIAN"].state_dict()
        }

        torch.save(ckpt, path)

    def load(self, path: Path=None) -> int:
        if path is None:
            path = self.model_path
        
        ckpt = torch.load(path, map_location=self.device)
        self.registrar.load_models(ckpt['encoder'])
        self.model.load_state_dict(ckpt['model'])
        try:
            self.optimizer["PEDESTRIAN"].load_state_dict(ckpt['optim_state'])
            epoch = ckpt["epoch"]
        except KeyError:
            epoch = 0
            
        return epoch
    
class CustomAutoEncoder(AutoEncoder):
    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        # dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        # predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_vel


    def get_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        
        feat_x_encoded = self.encode(batch,node_type) # B * 64
        loss = self.diffusion.get_loss(y_st_t.cuda(), feat_x_encoded)  # y_t -> y_st_t
        return loss