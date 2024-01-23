import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from trajdata.data_structures import AgentType
import warnings
warnings.simplefilter("ignore")

from keepflow.models import ModelTemplate
from keepflow.utils import EmptyEnv
sys.path.append('extern/traj/trajectron/trajectron')
from model import MultimodalGenerativeCVAE
from model.model_registrar import ModelRegistrar
from model.model_utils import ModeKeys
from model.dataset.preprocessing import restore
from model.components import GMM2D


class Trajectron(ModelTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        with open("extern/traj/trajectron/experiments/pedestrians/models/eth_vel/config.json", 'r', encoding='utf-8') as conf_json:
            hyperparams = json.load(conf_json)
            
        hyperparams['learning_rate'] = 0.01
        hyperparams['dynamic_edges'] = 'yes'
        hyperparams['edge_state_combine_method'] = 'sum'
        hyperparams['edge_influence_combine_method'] = 'attention'
        hyperparams['edge_addition_filter'] = [0.25, 0.5, 0.75, 1.0]
        hyperparams['edge_removal_filter'] = [1.0, 0.0]
        hyperparams['batch_size'] = cfg.DATA.BATCH_SIZE
        hyperparams['k_eval'] = 20
        hyperparams['offline_scene_graph'] = 'no'
        hyperparams['incl_robot_node'] = False
        hyperparams['node_freq_mult_train'] = False
        hyperparams['node_freq_mult_eval'] = False
        hyperparams['scene_freq_mult_train'] = False
        hyperparams['scene_freq_mult_eval'] = False
        hyperparams['scene_freq_mult_viz'] = False
        hyperparams['edge_encoding'] = True
        hyperparams['use_map_encoding'] = False
        hyperparams['augment'] = True
        hyperparams['override_attention_radius'] = []
                
        self.hyperparams = hyperparams

        self.curr_iter = 0
        self.ph = self.hyperparams['prediction_horizon']

        env = EmptyEnv(cfg)
                
        self.model_registrar = ModelRegistrar(cfg.SAVE_DIR, self.device)
        self.node_models_dict = dict()
        self.nodes = set()
        
        self.env = env
        self.set_environment(env)
        self.set_annealing_params()
        
        self.optimizer = dict()
        for node_type in env.PredNodeType:
            self.optimizer[node_type] = optim.Adam(
                [
                    {'params': self.model_registrar.get_all_but_name_match('map_encoder').parameters(), 'lr': hyperparams['learning_rate']},
                    {'params': self.model_registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008}
                ],)
            
        self.optimizers = self.optimizer.values()

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.PredNodeType:
            self.node_models_dict[node_type] = CustomMultimodalGenerativeCVAE(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            )

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    def train_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x_t,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y_t,
                                labels_st=y_st_t,
                                neighbors=neighbors_data_st,
                                neighbors_edge_value=neighbors_edge_value,
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph)

        return loss
    
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
        
        batch = (data_dict["first_history_index"],
                 data_dict["obs"],
                 data_dict["gt"],
                 data_dict["obs"],
                 data_dict["gt"],
                 neighbors,
                 neighbors_edge,
                 data_dict["robot_traj"],
                 data_dict["map"])
        
        return batch
    
    def predict(self, data_dict, return_prob=False):
        batch = self.create_batch(data_dict)
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        
        node_type = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')  # assume all agnet types are the same
        model = self.node_models_dict[node_type]
        y_dist, predictions = model.predict(inputs=x_t,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    neighbors=neighbors_data_st,
                                    neighbors_edge_value=neighbors_edge_value,
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.pred_len,
                                    num_samples=1,
                                    z_mode=False,
                                    gmm_mode=False,  # if sampling z from mean distribution # True for simfork
                                    full_dist=True,  # if sampling from all latent modes
                                    all_z_sep=False)
        
        data_dict[("pred", 0)] = predictions[0]
        if return_prob:
            data_dict[("gt_traj_log_prob", 0)] = y_dist.log_prob(data_dict["gt"]).squeeze(1)
            y_dist = model.dynamic.integrate_distribution(y_dist)
            data_dict[("prob", 0)] = y_dist
            y_dist.log_prob(torch.rand_like(predictions).tile(1, 10000, 1, 1))  # calculate the probability for the fair comparison on inference cost
        return data_dict           
        
    def predict_from_new_obs(self, data_dict, time_step: int):
        # do nothing
        return data_dict
    
    def update(self, data_dict):
        batch = self.create_batch(data_dict)
        
        node_type = str(AgentType(data_dict['agent_type'][0].item())).lstrip('AgentType.')  # assume all agnet types are the same in this minibatch
        
        self.step_annealers(node_type)
        self.optimizer[node_type].zero_grad()
        train_loss = self.train_loss(batch, node_type)
        train_loss.backward()
        self.optimizer[node_type].step()
        
        return {"loss": train_loss.item()}
    
    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.model_path
            
        ckpt = {
            'epoch': epoch,
            'model': self.model_registrar.state_dict(),
            'optim_state': {k: self.optimizer[k].state_dict() for k in self.optimizer}
        }

        torch.save(ckpt, path)
        

    def load(self, path: Path=None) -> int:
        if path is None:
            path = self.model_path
        
        ckpt = torch.load(path, map_location=self.device)
        self.model_registrar.load_state_dict(ckpt['model'])
        
        try:
            for node_type in self.env.PredNodeType:
                self.optimizer[node_type].load_state_dict(ckpt['optim_state'][node_type])
            epoch = ckpt["epoch"]
        except KeyError:
            epoch = 0

        return epoch


class CustomMultimodalGenerativeCVAE(MultimodalGenerativeCVAE):
    def predict(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                   inputs=inputs,
                                                                   inputs_st=inputs_st,
                                                                   labels=None,
                                                                   labels_st=None,
                                                                   first_history_indices=first_history_indices,
                                                                   neighbors=neighbors,
                                                                   neighbors_edge_value=neighbors_edge_value,
                                                                   robot=robot,
                                                                   map=map)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        y_dist, our_sampled_future = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode)

        return y_dist, our_sampled_future
    
    
    def p_y_xz(self, mode, x, x_nr_t, y_r, n_s_t0, z_stacked, prediction_horizon,
               num_samples, num_components=1, gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.node_type + '/decoder/state_action'](n_s_t0)

        state = initial_state
        if self.hyperparams['incl_robot_node']:
            input_ = torch.cat([zx,
                                a_0.repeat(num_samples * num_components, 1),
                                x_nr_t.repeat(num_samples * num_components, 1)], dim=1)
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(ph):
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, num_components))

            if self.hyperparams['incl_robot_node']:
                dec_inputs = [zx, a_t, y_r[:, j].repeat(num_samples * num_components, 1)]
            else:
                dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
        
        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)
        
        a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                       torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(corrs, [num_samples, -1, ph, num_components]))

        if self.hyperparams['dynamic'][self.node_type]['distribution']:
            y_dist = self.dynamic.integrate_distribution(a_dist, x)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            # sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, a_sample
        else:
            return y_dist
        
    def train_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        mask = ~(torch.isnan(labels).sum(dim=[1, 2]) > 0)
        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels.nan_to_num(),
                                                                     labels_st=labels_st.nan_to_num(),
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)

        z, kl = self.encoder(mode, x, y_e)
        log_p_y_xz = self.decoder(mode, x, x_nr_t, y, y_r, n_s_t0, z,
                                  labels.nan_to_num(),  # Loss is calculated on unstandardized label
                                  prediction_horizon,
                                  self.hyperparams['k'])
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0) * mask # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist)
        
        ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        loss = -ELBO

        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_y_xz'),
                                          log_p_y_xz_mean,
                                          self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_q'),
                                       mutual_inf_q,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_p'),
                                       mutual_inf_p,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'log_likelihood'),
                                       log_likelihood,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                       loss,
                                       self.curr_iter)
            if self.hyperparams['log_histograms']:
                self.latent.summarize_for_tensorboard(self.log_writer, str(self.node_type), self.curr_iter)
        return loss
    
def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()