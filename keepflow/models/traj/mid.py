import sys
import yaml
from pathlib import Path
from easydict import EasyDict
import torch
import torch.optim as optim
from trajdata.data_structures import AgentType

from keepflow.data.traj import EmptyEnv
from keepflow.models import ModelTemplate
sys.path.append('extern/traj/MID')
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
from models.trajectron import Trajectron
from models.encoders import MultimodalGenerativeCVAE
from models.autoencoder import AutoEncoder
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj, VarianceSchedule


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
                    
        self.registrar = ModelRegistrar(self.save_dir, self.device)
        
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        env = EmptyEnv(cfg)
        self.env = env
        self.encoder.set_environment(env)
        self.encoder.set_annealing_params()
        
        self.model = CustomAutoEncoder(self.config, encoder=self.encoder)
        
        self.optimizer = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                     {'params': self.model.parameters()}
                                     ],
                                    lr=self.config.lr)
        
        self.optimizers = [self.optimizer]
        
    def to(self, device):
        self.model.to(device)
        self.registrar.to(device)
        
    def eval(self):
        # self.model.eval()
        # self.registrar.eval()
        pass
        
    def train(self):
        # self.model.train()
        # self.registrar.train()
        pass
    
    def create_batch(self, data_dict):
        agent_type = data_dict['agent_type']
        neighbor_type = data_dict['neighbor_type']
        
        neighbors = {edge_type : [] for edge_type in self.env.EdgeType}
        neighbors_edge = {edge_type : [] for edge_type in self.env.EdgeType}
        
        for i in range(len(agent_type)):
            for edge_type in neighbors.keys():
                if AgentType[edge_type[0].name] == agent_type[i]:
                    idx = torch.where(neighbor_type[i] == AgentType[edge_type[1].name])
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
    
    def predict(self, data_dict, return_prob=False):
        batch = self.create_batch(data_dict)
        
        n_sample = self.kde_n_sample if return_prob else 20
        
        node_type = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')  # assume all agnet types are the same in this minibatch
        traj_pred = self.model.generate(batch, node_type, num_points=12, sample=n_sample, bestof=True)
        traj_pred = traj_pred.permute(1, 0, 2, 3)
        
        data_dict[("pred", 0)] = traj_pred[:, :20]
        
        if return_prob:
            traj_pred = torch.cat([traj_pred, torch.zeros_like(traj_pred)], dim=3)
            data_dict[("prob", 0)] = traj_pred[..., :3]
        
        return data_dict
    
    def training_step(self, data_dict):
        batch = self.create_batch(data_dict)
        
        node_type = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')  # assume all agnet types are the same in this minibatch
        self.optimizer.zero_grad()
        
        train_loss = self.model.get_loss(batch, node_type)
        train_loss.backward()
        self.optimizer.step()
        
        return {"loss": train_loss.item()}
    
    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.model_path

        ckpt = {
            'epoch': epoch,
            'encoder': self.registrar.model_dict,
            'model': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict()
        }

        torch.save(ckpt, path)

    def load(self, path: Path=None) -> int:
        if path is None:
            path = self.model_path
        
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.registrar.load_models(ckpt['encoder'])
        
        # craete new Trajectron with loaded weights
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        self.encoder.set_environment(self.env)
        self.encoder.set_annealing_params()
        
        # hand over Trajectron encoder with weights to ddpm model
        self.model = CustomAutoEncoder(self.config, encoder=self.encoder)
        self.model.to(self.device)
        self.model.load_state_dict(ckpt['model'])
        
        self.optimizer.load_state_dict(ckpt['optim_state'])
        epoch = ckpt["epoch"]
                    
        return epoch
    
class CustomAutoEncoder(AutoEncoder):
    def __init__(self, config, encoder):
        torch.nn.Module.__init__(self)
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = CustomDiffusionTraj(
            net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'
            )
        )
    
    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=1):
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
        loss = self.diffusion.get_loss(y_st_t, feat_x_encoded)  # y_t -> y_st_t
        return loss
    
    
class CustomDiffusionTraj(DiffusionTraj):
    def get_loss(self, x_0, context, t=None):

        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].to(x_0.device)

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).to(x_0.device)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).to(x_0.device)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0).to(x_0.device) # (B, N, d)

        mask = ~(torch.isnan(x_0).sum(dim=1, keepdim=True) > 0)
        e_theta = self.net(c0 * x_0.nan_to_num() + c1 * e_rand, beta=beta, context=context)
        loss = (e_theta - e_rand) ** 2 
        loss *= mask
        return loss.mean()
    
    def sample(self, num_points, context, sample, bestof, point_dim=2, flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        ## original: sampling traj sequentially
        ## changed to parallel sampling
        traj_list = []
        batch_size = context.size(0) * sample
        if bestof:
            x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        else:
            x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        stride = step
        # for t in range(self.var_sched.num_steps, 0, -1):
        for t in range(self.var_sched.num_steps, 0, -stride):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context.tile(sample, 1))
            if sampling == "ddpm":
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            elif sampling == "ddim":
                x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
        
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            traj_list.append(traj)
        else:
            traj_list.append(traj[0])
        
        return torch.stack(traj_list, dim=1).reshape([sample, context.size(0), num_points, point_dim])