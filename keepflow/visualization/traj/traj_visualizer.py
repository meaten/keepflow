from yacs.config import CfgNode
from typing import Dict, List, Tuple

from pathlib import Path
from joblib import delayed, Parallel
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import griddata, RBFInterpolator, interp2d

from keepflow.visualization import Visualizer
from keepflow.visualization import plot_density
from keepflow.data.traj.trajectron_dataset import get_hypers

class TrajVisualizer(Visualizer):
    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        
        self.device = cfg.DEVICE
        self.model_name = cfg.MODEL.TYPE
        
        self.dataset = cfg.DATA.DATASET_NAME
        
        import dill
        env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "processed_data" / f"{cfg.DATA.DATASET_NAME}_test.pkl"
        with open(env_path, 'rb') as f:
            self.env = dill.load(f, encoding='latin1')
        
        # TODO: params for other nodes
        hypers = get_hypers(cfg)
        node = 'PEDESTRIAN'
        self.state = hypers[cfg.DATA.TRAJ.PRED_STATE][node]
        mean, std = self.env.get_standardize_params(self.state, node)
        self.mean = mean
        self.std = std
        self.std = self.env.attention_radius[('PEDESTRIAN', 'PEDESTRIAN')]
        
        #self.num_grid = 500  # for visualization
        self.num_grid = 100
        if hasattr(self.env, "gt_dist"):
            self.gt_dist = self.env.gt_dist
        else:
            self.gt_dist = None
        
        self.observe_length = cfg.DATA.OBSERVE_LENGTH

    def __call__(self, dict_list: List[Dict]) -> None:
        index = dict_list[0]['index']
        min_pos, max_pos = self.get_minmax(index)
        
        # (batch, timesteps, [x,y])
        obs = self.to_numpy(dict_list[0]['obs'][:, :, 0:2])
        gt = self.to_numpy(dict_list[0]['gt'])

        pred = []
        for d in dict_list:
            pred.append(self.to_numpy(d[("pred", 0)][:, :, None]))
            assert np.all(obs == self.to_numpy(d["obs"][:, :, 0:2]))
            assert np.all(gt == self.to_numpy(d["gt"]))
        
        # (batch, timesteps, num_trials, [x,y])
        pred = np.concatenate(pred, axis=2)
        for i in range(len(obs)):
            self.plot2d_trajectories(obs[i:i+1],
                                     gt[i:i+1],
                                     pred[i:i+1],
                                     index[i],
                                     max_pos,
                                     min_pos)
        
        if ("prob", 0) in dict_list[0]:
            path_density_map = self.output_dir / "density_map"
            path_density_map.mkdir(exist_ok=True)
            
            xx, yy = self.get_grid(index)
            
            for k in dict_list[0].keys():
                if k[0] == "prob":
                    update_step = k[1]
                    prob = dict_list[0][k]
                    
                    bs, _, timesteps = prob.shape
                    
                    obs = self.to_numpy(dict_list[0]['obs'])[..., :2]
                    gt = self.to_numpy(dict_list[0]['gt'])
                    traj = np.concatenate([obs, gt], axis=1)
                    obs = traj[:, :self.observe_length + update_step]
                    gt = traj[:, self.observe_length + update_step-1:]
                    
                    zz_list = []
                    for j in range(timesteps):
                        zz = prob[0, :, j].reshape(xx.shape)
                        zz /= np.max(zz)
                        plot_density(xx, yy, zz, path=path_density_map / f"update{update_step}_{index[i][0]}_{index[i][1]}_{index[i][2].strip('PEDESTRIAN/')}_{j}.png",
                                   traj=[obs[i], gt[i]])
                        zz_list.append(zz)
                        
                    zz_sum = sum(zz_list)
                    plot_density(xx, yy, zz_sum, 
                                   path=path_density_map / f"update{update_step}_{index[i][0]}_{index[i][1]}_{index[i][2].strip('PEDESTRIAN/')}_sum.png",
                                   traj=[obs[i], gt[i]])
                            
    def prob_on_grid(self, dict_list: List[Dict]) -> List:
        if ("prob", 0) in dict_list[0]:
            index = dict_list[0]['index']
            min_pos, max_pos = self.get_minmax(index)
            xx, yy = self.get_grid(index)
            
            for data_dict in dict_list:
                data_dict["grid"] = [xx, yy]
                data_dict["minmax"] = [min_pos, max_pos]
                for k in list(data_dict.keys()):
                    if k[0] == "prob":
                        prob = data_dict[k]
                        if type(prob) == torch.Tensor:
                            prob = self.to_numpy(prob)
                            batch, _, timesteps, _ = prob.shape
                        
                            zz_batch = []                          
                            for i in range(batch):
                                zz_timesteps = Parallel(n_jobs=timesteps)(delayed(self.griddata_on_cluster)(i, prob, xx, yy, max_pos, min_pos, j)
                                                                    for j in range(timesteps))
                                # zz_timesteps = [self.griddata_on_cluster(i, prob, xx, yy, max_pos, min_pos, j) for j in range(timesteps)]
                                zz_timesteps = np.stack(zz_timesteps, axis=-1).reshape(-1, timesteps)
                                zz_batch.append(zz_timesteps)
                            
                            zz_batch = np.stack(zz_batch, axis=0)
                            data_dict[k] = zz_batch
                            
                        elif self.model_name == "Trajectron" or self.model_name == "GT_Dist":
                            value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[None, :, None].tile(1, 1, prob.mus.shape[2], 1).to(self.device)
                            zz_batch = self.to_numpy(torch.exp(prob.log_prob(value)))
                            data_dict[k] = zz_batch
                            
                        else:
                            zz_batch = []
                            for i in range(len(prob)):
                                zz_timesteps = []
                                for kernel in prob[i]:
                                    zz = kernel(torch.Tensor(np.array([xx.flatten(), yy.flatten()]).T).to(self.device))
                                    zz_timesteps.append(zz.cpu().numpy())
                                zz_timesteps = np.stack(zz_timesteps, axis=1)
                                zz_batch.append(zz_timesteps)
                            zz_batch = np.stack(zz_batch, axis=0)
                            data_dict[k] = zz_batch
                        
                        if ("gt_traj_log_prob", k[1]) not in data_dict:
                            gt = data_dict["gt"][:, k[1]:].cpu()
                            timesteps = gt.shape[1]
                            
                            gt_traj_prob = np.array([interp2d(xx, yy, zz_batch[:, :, t])(gt[:, t, 0], gt[:, t, 1]) for t in range(timesteps)])
                            gt_traj_log_prob = torch.log(torch.Tensor(gt_traj_prob)).squeeze()
                            if torch.sum(torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)) > 0:
                                mask = torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)
                                value = torch.min(gt_traj_log_prob[~mask])
                                gt_traj_log_prob = torch.nan_to_num(gt_traj_log_prob, nan=value, neginf=value)
                            data_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob[None]
                        
                if self.gt_dist is not None: #assume simfork
                    bs, timesteps, d = data_dict["gt"].shape
                    split = True
                    #split = False
                    value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[None, :, None].tile(1, 1, timesteps, 1).to(self.device)
                    if split:
                        gaussians = []
                        for i in range(len(self.gt_dist)):
                            gt_base_traj = torch.Tensor(self.gt_dist[[i] * bs, -timesteps:]).to(self.device)
                            gaussians.append(Normal(gt_base_traj[..., :2], gt_base_traj[..., 2:]))
                        data_dict["gt_prob"] = sum([torch.exp(g.log_prob(value).sum(dim=-1)) for g in gaussians])
                    else:
                        gt_base_traj = torch.Tensor(self.gt_dist[data_dict["gt"][:, -1, 1] < 0, -timesteps:]).to(self.device)
                        gaussian = Normal(gt_base_traj[:, :2], gt_base_traj[:, 2:])
                        data_dict["gt_prob"] = torch.exp(gaussian.log_prob(value).sum(dim=-1))
                    
        return dict_list
    
    def get_grid(self, index):
        min_pos, max_pos = self.get_minmax(index)
        xs = np.linspace(min_pos[0], max_pos[0], num=self.num_grid)
        ys = np.linspace(min_pos[1], max_pos[1], num=self.num_grid)
        xx, yy = np.meshgrid(xs, ys)
        
        return xx, yy

    def get_minmax(self, index):
        idx = [s.name for s in self.env.scenes].index(index[0][0])
        max_pos, min_pos = self.env.scenes[idx].calculate_pos_min_max()
        max_pos += 0.05 * (max_pos - min_pos)
        min_pos -= 0.05 * (max_pos - min_pos)
        return min_pos, max_pos
                    
    def griddata_on_cluster(self, i, prob, xx, yy, max_pos, min_pos, j):
        prob_ = prob[i, :, j]
        prob_ = prob_[np.where(np.isinf(prob_).sum(axis=1) == 0)]
        prob_ = prob_[np.where(np.isnan(prob_).sum(axis=1) == 0)]
        lnk = linkage(prob_[:, :-1],
                        method='single',
                        metric='euclidean')
        idx_cls = fcluster(lnk, t=np.linalg.norm(max_pos-min_pos)*0.003, 
                            criterion='distance')
        idx_cls -= 1
        
        zz_ = []
        """
        xs = np.linspace(min_pos[0], max_pos[0], num=300)
        ys = np.linspace(min_pos[1], max_pos[1], num=300)
        
        upper = np.array([xs, np.ones_like(xs) * min_pos[1], np.ones_like(xs) * 0.0]).T
        bottom = np.array([xs, np.ones_like(xs) * max_pos[1], np.ones_like(xs) * 0.0]).T
        right = np.array([np.ones_like(ys) * max_pos[0], ys, np.ones_like(ys) * 0.0]).T
        left = np.array([np.ones_like(ys) * min_pos[0], ys, np.ones_like(ys) * 0.0]).T
        boundary = np.concatenate([upper, bottom, right, left], axis=0)
        prob_ = np.concatenate([prob[i, :, j], boundary], axis=0)
        
        
        intp = RBFInterpolator(prob_[:, :-1],
                               prob_[:, -1],
                               smoothing=10, kernel='linear')
        zz__ = intp(np.array([xx, yy]).transpose().reshape(-1, 2))
        zz_.append(zz__.reshape(xx.shape).T)
        """
        for c in range(np.max(idx_cls)+1):
            try:
                zz_.append(griddata(prob_[idx_cls == c, :-1],
                                prob_[idx_cls == c, -1],
                                (xx, yy), method='linear',
                                fill_value=0.0)
                           )
            except:
                pass
        
        zz = sum(zz_)
        
        intp = RBFInterpolator(np.array([xx, yy]).reshape(2, -1).T, zz.flatten(), smoothing=10, kernel='linear', neighbors=8)
        zz = intp(np.array([xx, yy]).reshape(2, -1).T).reshape(xx.shape)
        
        prob_ = prob[i, :, j, -1]
        prob_ = prob_[~np.isnan(prob_)]
        zz = np.clip(zz, a_min=0.0, a_max=np.percentile(prob_, 95))
        
        
        #zz /= np.sum(zz)
        
        #plot_density(
        #    xx, yy, zz, path=path_density_map / f"update{update_step}_{index[0]}_{index[1]}_{index[2].strip('PEDESTRIAN/')}_{j}.png",
        #    traj=traj)
        return zz

    def plot2d_trajectories(self,
                            obs:  np.ndarray,
                            gt:   np.ndarray,
                            pred: np.ndarray,
                            index: Tuple,
                            max_pos: np.ndarray,
                            min_pos: np.ndarray) -> None:
        """plot 2d trajectories

        Args:
            obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
            gt (np.ndarray): (N_seqs, N_timesteps, [x,y])
            pred (np.ndarray): (N_seqs, N_timesteps, N_trials, [x,y])
            img_path (Path): Path
        """

        N_seqs, N_timesteps, N_trials, N_dim = pred.shape
        gt_vis = np.zeros([N_seqs, N_timesteps+1, N_dim])
        gt_vis[:, 0] = obs[:, -1]
        gt_vis[:, 1:] = gt

        pred_vis = np.zeros([N_seqs, N_timesteps+1, N_trials, N_dim])
        # (num_seqs, num_dim) -> (num_seqs, 1, num_dim)
        pred_vis[:, 0] = obs[:, -1][:, None]
        pred_vis[:, 1:] = pred

        f, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_pos[0], max_pos[0])
        ax.set_ylim(min_pos[1], max_pos[1])

        for j in range(N_seqs):
            sns.lineplot(x=obs[j, :, 0], y=obs[j, :, 1], color='black',
                         legend='brief', label="obs", marker='o')
            sns.lineplot(x=gt_vis[j, :, 0], y=gt_vis[j, :, 1],
                         color='blue', legend='brief', label="GT", marker='o')
            for i in range(pred.shape[2]):
                if i == 0:
                    sns.lineplot(x=pred_vis[j, :, i, 0], y=pred_vis[j, :, i, 1],
                                 color='green', legend='brief', label="pred", marker='o')
                else:
                    sns.lineplot(
                        x=pred_vis[j, :, i, 0], y=pred_vis[j, :, i, 1], color='green', marker='o')
                    
        img_path = self.output_dir / f"{index[0]}_{index[1]}_{index[2].strip('PEDESTRIAN/')}.png"
        plt.savefig(img_path)
        plt.close()
        
        
def plot2d_trajectories_samples(
                        obs:  np.ndarray,
                        gt:   np.ndarray,
                        max_pos: np.ndarray,
                        min_pos: np.ndarray) -> None:
    """plot 2d trajectories

    Args:
        obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
        gt (np.ndarray): (N_seqs, N_timesteps, [x,y])
        pred (np.ndarray): (N_seqs, N_timesteps, N_trials, [x,y])
        img_path (Path): Path
    """

    N_seqs = len(gt)
    _, N_timesteps, N_dim = gt[0].shape
    
    f, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_pos[0], max_pos[0])
    ax.set_ylim(min_pos[1], max_pos[1])

    for j in range(N_seqs):
        gt_vis = np.zeros([N_timesteps+1, N_dim])
        gt_vis[0] = obs[j][0, -1]
        gt_vis[1:] = gt[j][0]
        if j == 0:
            sns.lineplot(x=obs[j][0, :, 0], y=obs[j][0, :, 1], color='green',
                            legend='brief', label="obs", marker='o')
            sns.lineplot(x=gt_vis[:, 0], y=gt_vis[:, 1],
                            color='black', legend='brief', label="GT", marker='o')
        else:
            sns.lineplot(x=obs[j][0, :, 0], y=obs[j][0, :, 1], color='green',
                            marker='o')
            sns.lineplot(x=gt_vis[:, 0], y=gt_vis[:, 1],
                            color='black', marker='o')
                
    img_path = "gts.png"
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
    

    




