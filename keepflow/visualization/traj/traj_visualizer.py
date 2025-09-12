from yacs.config import CfgNode
from typing import Dict, List, Tuple

from pathlib import Path
from joblib import delayed, Parallel
import numpy as np
import torch
import torch.distributions as D
import seaborn as sns
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator, LinearNDInterpolator

from keepflow.data.traj import EmptyEnv
from keepflow.visualization import Visualizer
from keepflow.visualization import plot_density
from keepflow.utils.kde import GaussianKDE


class TrajVisualizer(Visualizer):
    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        
        self.device = cfg.DEVICE
        self.model_name = cfg.MODEL.TYPE
        
        self.dataset = cfg.DATA.DATASET_NAME
        
        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len = cfg.DATA.PREDICT_LENGTH
        
        self.num_grid = 500  # for detailed visualization
        # self.num_grid = 100
        
        if 'sim' in self.dataset:
            import dill
            env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "processed_data" / f"{cfg.DATA.DATASET_NAME}_test.pkl"
            with open(env_path, 'rb') as f:
                self.env = dill.load(f, encoding='latin1')
                self.gt_dist_type = self.env.gt_dist_type
                self.gt_dist = self.env.gt_dist
                
        else:
            self.env = EmptyEnv(cfg)
            self.gt_dist = None

    def __call__(self, result_dict: Dict) -> None:
        # assume batch size is 1
        index = result_dict['index']
        min_pos, max_pos = self.get_minmax(result_dict)
        
        # (batch, timesteps, [x,y])
        obs = self.to_numpy(result_dict['obs'][:, :, 0:2])
        gt = self.to_numpy(result_dict['gt'])
        
        if len(result_dict['neighbors']) == 0:
            neighbor = np.zeros([len(obs), 0, 1, 2])
            neighbor_gt = np.zeros([len(obs), 0, 1, 2])
        else:
            neighbor = [self.to_numpy(n) for n in result_dict['neighbors']]
            neighbor_gt = [self.to_numpy(n) for n in result_dict['neighbors_gt']]
        
        pred = self.to_numpy(result_dict[("pred", 0)])
        
        for i in range(len(obs)):
            plot2d_trajectories(
                self.output_dir,
                obs[i],
                gt[i],
                pred[i],
                neighbor[i],
                neighbor_gt[i],
                index[i],
                max_pos,
                min_pos)
        
        if ("prob_grid", 0) in result_dict:
            path_density_map = self.output_dir / "density_map"
            path_density_map.mkdir(exist_ok=True)
            
            xx, yy = get_grid(self.num_grid, min_pos, max_pos)
            
            for k in result_dict.keys():
                if k[0] == "prob_grid":
                    update_step = k[1]
                    prob = result_dict[k]
                    
                    bs, _, timesteps = prob.shape
                    
                    obs = self.to_numpy(result_dict['obs'])[..., :2]
                    gt = self.to_numpy(result_dict['gt'])
                    traj = np.concatenate([obs, gt], axis=1)
                    obs = traj[:, :self.obs_len + update_step]
                    gt = traj[:, self.obs_len + update_step-1:]
                    
                    split = index[i][0]
                    timestep = int(index[i][1])
                    agent_id = int(float(index[i][2].strip('PEDESTRIAN/')))
                    
                    zz_list = []
                    for j in range(timesteps):
                        zz = prob[0, :, j].reshape(xx.shape)
                        zz /= np.max(zz)
                        plot_density(xx, yy, zz, path=path_density_map / f"update{update_step}_{split}_{timestep:03}_{agent_id:03}_{j:02}.png",
                                   traj=[obs[i], gt[i]])
                        zz_list.append(zz)
                        
                    zz_sum = sum(zz_list)
                    plot_density(xx, yy, zz_sum, 
                                   path=path_density_map / f"update{update_step}_{split}_{timestep:03}_{agent_id:03}_sum.png",
                                   traj=[obs[i], gt[i]])
                            
    def prob_on_grid(self, result_dict: Dict) -> Dict:
        if ("prob", 0) in result_dict:
            index = result_dict['index']
            min_pos, max_pos = self.get_minmax(result_dict)
            xx, yy = get_grid(self.num_grid, min_pos, max_pos)
            
            result_dict["grid"] = [xx, yy]
            result_dict["minmax"] = [min_pos, max_pos]
            for k in list(result_dict.keys()):
                if k[0] == "prob":
                    prob = result_dict[k]
                    
                    # samples with probability
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
                        result_dict[('prob_grid', k[1])] = zz_batch
                        
                    # gaussian mixture model
                    elif self.model_name == "Trajectron":
                        query = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[None, :, None].tile(1, 1, prob.mus.shape[2], 1).to(self.device)
                        zz_batch = self.to_numpy(torch.exp(prob.log_prob(query)))
                        result_dict[('prob_grid', k[1])] = zz_batch
                        
                    # kernel density estimation
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
                        result_dict[('prob_grid', k[1])] = zz_batch
                    
                    if ("gt_traj_log_prob", k[1]) not in result_dict:
                        gt = result_dict["gt"][:, k[1]:].cpu()
                        timesteps = gt.shape[1]
                        points = np.stack([xx.flatten(), yy.flatten()], axis=1)
                        gt_traj_prob = []
                        for i in range(len(zz_batch)):
                            gt_traj_prob_ = np.array([LinearNDInterpolator(points, zz_batch[i, :, t])(gt[i, t, 0], gt[i, t, 1]) for t in range(timesteps)])
                            gt_traj_prob.append(gt_traj_prob_)
                        gt_traj_prob = np.stack(gt_traj_prob, axis=0)
                        gt_traj_log_prob = torch.log(torch.Tensor(gt_traj_prob)).squeeze()
                        if torch.sum(torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)) > 0:
                            mask = torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)
                            try:
                                min_log_prob = torch.min(gt_traj_log_prob[~mask])
                            except:
                                min_log_prob = -100
                            gt_traj_log_prob = torch.nan_to_num(gt_traj_log_prob, nan=min_log_prob, neginf=min_log_prob)
                        result_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob[None]
                
                    if self.gt_dist is not None: #assume simulated dataset, for EMD metric
                        bs, timesteps, d = result_dict["gt"].shape
                        step = k[1]
                        if step == 0:
                            if self.gt_dist_type == 'gmm':
                                query = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[None, :, None].tile(1, 1, timesteps, 1)
                                gt_log_prob = self.gt_dist.log_prob(query)
                                gt_prob = torch.exp(gt_log_prob)
                            elif self.gt_dist_type == 'kde':
                                query = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1).cuda()
                                gt_prob = []
                                for i in range(timesteps):
                                    gt_prob.append(self.gt_dist[i](query))
                                gt_prob = torch.stack(gt_prob, dim=1)[None]
                        else:
                            if self.gt_dist_type == 'gmm':
                                query = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[None, :, None].tile(1, 1, timesteps, 1)
                                means = self.gt_dist.component_distribution.mean
                                stds = torch.sqrt(self.gt_dist.component_distribution.variance)
                                _, N_modes, _ = means.shape
                                costs = (means[:step] - gt[0, :step, None]).norm(dim=-1).sum(dim=0)
                                idxs = torch.argsort(costs)[:N_modes // 2]
                                mix = D.Categorical(torch.ones(N_modes // 2))
                                comp = D.Independent(D.Normal(means[step:, idxs], stds[step:, idxs]), 1)
                                gt_dist_updated = D.MixtureSameFamily(mix, comp)
                                gt_log_prob = gt_dist_updated.log_prob(query[:, :, step:])
                                gt_prob = torch.exp(gt_log_prob)
                            elif self.gt_dist_type == 'kde':
                                query = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1).cuda()
                                num_samples = 300
                                traj = torch.stack([torch.matmul(self.gt_dist[i].dataset, self.gt_dist[i].bw_transform.inverse())
                                                    for i in range(step)], dim=1)
                                costs = (traj.cpu() - gt[[0], :step]).norm(dim=-1).sum(dim=1)
                                idxs = torch.argsort(costs)[:num_samples]
                                gt_dist_updated = [GaussianKDE(torch.matmul(self.gt_dist[i].dataset[idxs], self.gt_dist[i].bw_transform.inverse()))
                                                for i in range(step, timesteps)]
                                gt_prob = torch.stack([gt_dist_updated[i](query) for i in range(timesteps - step)], dim=1)[None].cpu()
                        result_dict[("gt_prob", k[1])] = gt_prob
                        
        return result_dict

    def get_minmax(self, data_dict):
        if 'sim' in self.dataset:
            idx = [s.name for s in self.env.scenes].index(data_dict['index'][0][0])
            max_pos, min_pos = self.env.scenes[idx].calculate_pos_min_max()
        else:
            obs = data_dict['obs'][..., 0:2]
            gt = data_dict['gt']
            agent = torch.cat([obs, gt], dim=1)
            all_traj = agent[:, None]
            if len(data_dict['neighbors']) != 0:  # if neighbors exist
                neighbor = data_dict['neighbors'][..., 0:2]
                neighbor_gt = data_dict['neighbors_gt']
                neighbor = torch.cat([neighbor, neighbor_gt], dim=2)
                all_traj = torch.cat([all_traj, neighbor], dim=1)
            
            
            k = ('prob', 0)
            if k in data_dict:
                traj_prob = data_dict[k][:, :, :, :-1]
                traj_prob = torch.cat([obs[:, None].expand(-1, traj_prob.shape[1], -1, -1), traj_prob], dim=2)
                all_traj = torch.cat([all_traj, traj_prob], dim=1)
                
            min_pos = torch.amin(all_traj.nan_to_num(all_traj.nanmean()), dim=(0, 1, 2)).cpu().numpy()
            max_pos = torch.amax(all_traj.nan_to_num(all_traj.nanmean()), dim=(0, 1, 2)).cpu().numpy()
            
        if np.max(max_pos - min_pos) > 5.0:
            max_pos += 0.05 * (max_pos - min_pos)
            min_pos -= 0.05 * (max_pos - min_pos)
        else:
            center = (max_pos + min_pos) / 2
            max_pos = np.array([2.5, 2.5]) + center
            min_pos = np.array([-2.5, -2.5]) + center
        
        return min_pos, max_pos
                    
    def griddata_on_cluster(self, i, prob, xx, yy, max_pos, min_pos, j):
        prob_ = prob[i, :, j]
        prob_ = prob_[np.where(np.isinf(prob_).sum(axis=1) == 0)]
        prob_ = prob_[np.where(np.isnan(prob_).sum(axis=1) == 0)]
        
        zz = griddata(prob_[..., :-1],
                    prob_[..., -1],
                    (xx, yy), method='linear',
                    fill_value=0.0)
    
        # avoid extrapolation
        from scipy.spatial import KDTree
        threshold_pixels = 0.5
        threshold_meter = np.linalg.norm(max_pos - min_pos) / np.mean(xx.shape) * threshold_pixels
        
        mask = np.stack([xx, yy], axis=-1).reshape([-1, 2])
        tree = KDTree(prob_[..., :-1])
        mask = tree.query(mask)[0] < threshold_meter
        mask = mask.reshape(xx.shape)
        zz = mask * zz
        
        intp = RBFInterpolator(
            np.array([xx, yy]).reshape(2, -1).T, zz.flatten(),
            smoothing=1, kernel='linear', neighbors=8)
        
        zz = intp(np.array([xx, yy]).reshape(2, -1).T).reshape(xx.shape)
        
        prob_ = prob[i, :, j, -1]
        prob_ = prob_[~np.isnan(prob_)]
        zz = np.clip(zz, a_min=0.0, a_max=np.percentile(prob_, 95))
        
        return zz
    

def get_grid(num_grid, min_pos, max_pos):
    xs = np.linspace(min_pos[0], max_pos[0], num=num_grid)
    ys = np.linspace(min_pos[1], max_pos[1], num=num_grid)
    xx, yy = np.meshgrid(xs, ys)
    
    return xx, yy
    

def plot2d_trajectories(output_dir: Path,
                        obs:  np.ndarray,
                        gt:   np.ndarray,
                        pred: np.ndarray,
                        neighbor: np.ndarray,
                        neighbor_gt: np.ndarray,
                        index: Tuple,
                        max_pos: np.ndarray,
                        min_pos: np.ndarray) -> None:
    """plot 2d trajectories

    Args:
        obs (np.ndarray): (N_timesteps, [x, y])
        gt (np.ndarray): (N_timesteps, [x,y])
        pred (np.ndarray): (N_timesteps, N_trials, [x,y])
        neighbor: (N_timesteps, N_neighbors, [x,y])  you can pass empty list [] for ignoring
        neighbor_gt: (N_timesteps, N_neighbors, [x,y])  you can pass empty list [] for ignoring
        index: [dataset_name, timestamp, agent_id]
        max_pos: [x,y]
        min_pos: [x,y]
    """

    N_trials, N_timesteps, N_dim = pred.shape
    N_neighbors, *_ = neighbor.shape
    
    f, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_pos[0], max_pos[0])
    ax.set_ylim(min_pos[1], max_pos[1])
    
    gt_vis = np.zeros([N_timesteps+1, N_dim])
    gt_vis[0] = obs[-1]
    gt_vis[1:] = gt

    sns.lineplot(x=obs[:, 0], y=obs[:, 1], color='black',
                    legend='brief', label="obs", marker='o', sort=False)
    sns.lineplot(x=gt_vis[:, 0], y=gt_vis[:, 1],
                    color='blue', legend='brief', label="GT", marker='o', sort=False)
    
    if not len(neighbor) == 0:
        neighbor_gt_vis = np.zeros([N_neighbors, N_timesteps+1, N_dim])
        neighbor_gt_vis[:, 0] = neighbor[:, -1]
        neighbor_gt_vis[:, 1:] = neighbor_gt
        
        for j in range(N_neighbors):
            sns.lineplot(x=neighbor[j, :, 0], y=neighbor[j, :, 1], color='black', marker='o', alpha=0.3, sort=False)
            sns.lineplot(x=neighbor_gt_vis[j, :, 0], y=neighbor_gt_vis[j, :, 1], color='blue', marker='o', alpha=0.3, sort=False)
    
    pred_vis = np.zeros([N_trials, N_timesteps+1, N_dim])
    # (num_seqs, num_dim) -> (num_seqs, 1, num_dim)
    pred_vis[:, 0] = obs[None, -1]
    pred_vis[:, 1:] = pred
    
    for i in range(N_trials):
        if i == 0:
            sns.lineplot(x=pred_vis[i, :, 0], y=pred_vis[i, :, 1],
                            color='green', legend='brief', label="pred", marker='o', sort=False)
        else:
            sns.lineplot(
                x=pred_vis[i, :, 0], y=pred_vis[i, :, 1], color='green', marker='o', sort=False)
            
    img_path = output_dir / f"{index[0]}_{index[2].strip('PEDESTRIAN/')}_{index[1]}.png"
    plt.savefig(img_path)
    plt.close()

    




