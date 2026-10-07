from yacs.config import CfgNode
from typing import Dict, List, Tuple

from pathlib import Path
from joblib import delayed, Parallel
import os
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
        min_pos, max_pos = self.get_minmax(result_dict, visualize=True)
        
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
    
            grid_size =  int(np.sqrt(result_dict[('prob_grid', 0)].shape[1]))
            xx, yy = get_grid(grid_size, min_pos, max_pos)
            
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
                        zz = zz / np.nanmax(zz)
                        # gt[i] starts at the last revealed position, so index j+1 is the
                        # absolute step this panel scores (t + update_step + j + 1).
                        pt = gt[i][j + 1] if j + 1 < len(gt[i]) else None
                        plot_density(xx, yy, zz, path=path_density_map / f"update{update_step}_{split}_{timestep:03}_{agent_id:03}_{j:02}.png",
                                   traj=[obs[i], gt[i]], gt_point=pt)
                        zz_list.append(zz)
                        
                    zz_sum = np.nansum(np.stack(zz_list, axis=0), axis=0)
                    plot_density(xx, yy, zz_sum, 
                                   path=path_density_map / f"update{update_step}_{split}_{timestep:03}_{agent_id:03}_sum.png",
                                   traj=[obs[i], gt[i]])
                            
    def prob_on_grid(self, result_dict: Dict, grid_size: int, visualize=False,
                      analytic_gt_log_prob=None) -> Dict:
        """
        Args:
            analytic_gt_log_prob: optional precomputed (batch, pred_len) tensor
                giving the ground-truth log-probability at step 0 directly from
                a model with an analytic density (see
                ``FlowChainTraj.analytic_log_prob``), bypassing the grid
                rasterisation + ``LinearNDInterpolator`` approximation below
                (and its -100 clamp outside the particle cloud's convex hull)
                for ``("gt_traj_log_prob", 0)``. Must be computed by the
                caller BEFORE ``metrics.denormalize`` mutates
                ``result_dict['gt']``, since it has to be evaluated in the
                model's own (pre-denormalize) coordinate space. None (default)
                preserves the legacy grid-interpolation behavior exactly.
        """
        if ("prob", 0) in result_dict:
            index = result_dict['index']
            min_pos, max_pos = self.get_minmax(result_dict, visualize=visualize)
            xx, yy = get_grid(grid_size, min_pos, max_pos)
            
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
                            zz_timesteps = Parallel(n_jobs=timesteps)(delayed(self.griddata_on_cluster)(i, prob, xx, yy, max_pos, min_pos, j,
                                                                                                         absent_as_nan=visualize)
                                                                for j in range(timesteps))
                            # zz_timesteps = [self.griddata_on_cluster(i, prob, xx, yy, max_pos, min_pos, j) for j in range(timesteps)]
                            zz_timesteps = np.stack(zz_timesteps, axis=-1).reshape(-1, timesteps)
                            zz_batch.append(zz_timesteps)
                        
                        zz_batch = np.stack(zz_batch, axis=0)
                        result_dict[('prob_grid', k[1])] = zz_batch
                        
                    # gaussian mixture model
                    elif self.model_name == "Trajectron":
                        query = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[None, :, None].tile(1, 1, prob.mus.shape[2], 1).to(self.device)
                        prob.mus += result_dict[('standardize_offset')][..., :2]
                        # The GMM can be evaluated at the ground truth directly,
                        # like the KDE baselines are, instead of being rasterised
                        # and interpolated. Same principle for every method: score
                        # each one on its own density, exactly.
                        _gt = result_dict["gt"][:, k[1]:].to(self.device)
                        _q = _gt[:, :, None].tile(1, 1, prob.mus.shape[2], 1)
                        result_dict[("gt_traj_log_prob", k[1])] = \
                            prob.log_prob(_q).diagonal(dim1=-2, dim2=-1).cpu() \
                            if prob.log_prob(_q).dim() > 2 else prob.log_prob(_q).cpu()
                        zz_batch = np.concatenate(
                            [self.to_numpy(torch.exp(prob.log_prob(query[:, i*10000:(i+1)*10000])))
                             for i in range(query.shape[1] // 10000 + 1)],
                            axis=1
                        )
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
                        _analytic_k = None
                        if isinstance(analytic_gt_log_prob, dict):
                            _analytic_k = analytic_gt_log_prob.get(k[1])
                            if _analytic_k is not None:
                                # The chained evaluation defines the density only
                                # from the update step onward; earlier steps are
                                # NaN and are dropped here so the metric averages
                                # over the steps that exist.
                                _analytic_k = _analytic_k[:, k[1]:]
                        elif analytic_gt_log_prob is not None and k[1] == 0:
                            _analytic_k = analytic_gt_log_prob
                        if _analytic_k is not None:
                            # Model evaluated analytically at the GT position --
                            # no grid rasterisation / interpolation / -100 clamp.
                            # See prob_on_grid's docstring and
                            # FlowChainTraj.analytic_log_prob for the contract
                            # (must be computed pre-denormalize by the caller).
                            gt_traj_log_prob = _analytic_k.detach().cpu()
                            result_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob
                        else:
                            gt = result_dict["gt"][:, k[1]:].cpu()
                            timesteps = gt.shape[1]
                            points = np.stack([xx.flatten(), yy.flatten()], axis=1)
                            gt_traj_prob = []
                            for i in range(len(zz_batch)):
                                gt_traj_prob_ = np.array([LinearNDInterpolator(points, zz_batch[i, :, t])(gt[i, t, 0], gt[i, t, 1]) for t in range(timesteps)])
                                gt_traj_prob.append(gt_traj_prob_)
                            gt_traj_prob = np.stack(gt_traj_prob, axis=0)
                            gt_traj_log_prob = torch.log(torch.Tensor(gt_traj_prob)).squeeze(0)
                            if torch.sum(torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)) > 0:
                                mask = torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)
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
                            elif self.gt_dist_type in ('kde', 'analytic'):
                                # 'analytic' supplies one exact density per step
                                # (keepflow/data/traj/analytic_density.py) with the
                                # same __call__ signature as GaussianKDE, so the
                                # marginal path is shared.
                                query = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1).cuda()
                                gt_prob = []
                                for i in range(timesteps):
                                    gt_prob.append(self.gt_dist[i](query))
                                gt_prob = torch.stack(gt_prob, dim=1)[None]
                        elif ('prob_grid', step) in result_dict:
                            if self.gt_dist_type == 'gmm':
                                query = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[None, :, None].tile(1, 1, timesteps, 1)
                                means = self.gt_dist.component_distribution.mean
                                stds = torch.sqrt(self.gt_dist.component_distribution.variance)
                                _, N_modes, _ = means.shape
                                # Exact posterior over mixture components given the
                                # observed prefix:
                                #   P(k | x_{1:step}) ∝ pi_k * prod_t N(x_t; mu_{t,k}, s_{t,k})
                                # The previous rule kept the `N_modes // 2` closest
                                # components regardless of `step`, which never narrows
                                # further as more observations arrive. For fork
                                # (2 modes -> 1) that happens to be right, but for
                                # branch (4 modes -> always 2) the conditional ground
                                # truth stays twice as wide as it should be once the
                                # second split is resolved, making a correctly
                                # narrowed prediction look worse in EMD.
                                obs_prefix = gt[0, :step, None]                      # (step, 1, 2)
                                log_lik = D.Independent(
                                    D.Normal(means[:step], stds[:step]), 1
                                ).log_prob(obs_prefix.expand(-1, N_modes, -1)).sum(0)  # (N_modes,)
                                log_w = torch.log_softmax(log_lik, dim=0)
                                mix = D.Categorical(logits=log_w)
                                comp = D.Independent(D.Normal(means[step:], stds[step:]), 1)
                                gt_dist_updated = D.MixtureSameFamily(mix, comp)
                                gt_log_prob = gt_dist_updated.log_prob(query[:, :, step:])
                                gt_prob = torch.exp(gt_log_prob)
                            elif self.gt_dist_type == 'analytic':
                                # Conditional (post-update) ground truth: keep the
                                # trajectories closest to the observed prefix, same
                                # semantics as the 'kde' branch below, but drawn
                                # from the exact generator instead of being limited
                                # to the 1000 trajectories in the test split.
                                from keepflow.data.traj.analytic_density import sample_unicycle
                                query = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1).cuda()
                                num_samples = 300
                                if getattr(self, "_analytic_samples", None) is None:
                                    self._analytic_samples = torch.as_tensor(
                                        sample_unicycle(200000, pred_len=timesteps, seed=0),
                                        dtype=torch.float32)
                                samples = self._analytic_samples
                                costs = (samples[:, :step] - gt[[0], :step].cpu()).norm(dim=-1).sum(dim=1)
                                idxs = torch.argsort(costs)[:num_samples]
                                gt_dist_updated = [GaussianKDE(samples[idxs, i].cuda())
                                                   for i in range(step, timesteps)]
                                gt_prob = torch.stack([gt_dist_updated[i](query)
                                                       for i in range(timesteps - step)], dim=1)[None].cpu()
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
                        result_dict[("gt_prob_grid", k[1])] = gt_prob
                        if self.gt_dist_type == 'gmm':
                            result_dict[("gt_prob_grid_exact", k[1])] = \
                                self._exact_gt_grid(result_dict, xx, yy, k[1])
                        
        return result_dict

    def _exact_gt_grid(self, result_dict, xx, yy, step):
        """Exact fork/branch posterior on the grid (keepflow/data/traj/sim_gt.py),
        conditioned on the observation window and, for an update from t+step,
        on the first `step` ground-truth positions. Shape (1, G*G, T-step)."""
        from keepflow.data.traj.sim_gt import ExactForkBranchGT
        if getattr(self, "_exact_gt", None) is None:
            self._exact_gt = ExactForkBranchGT(self.gt_dist, obs_len=self.obs_len)
        obs = result_dict["obs"][0, :, :2].detach().double().cpu()
        assert obs[-1].abs().max() < 1e-4, "expected the frame centred on the last observation"
        gt = result_dict["gt"][0].detach().double().cpu()
        self._exact_gt.condition(obs, gt[:step] if step > 0 else None)
        query = np.stack([xx.flatten(), yy.flatten()], axis=1)
        return self._exact_gt.grid_density(query)[None].float()

    def get_minmax(self, data_dict, visualize=False):
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
            
            if not visualize:
                k = ('pred', 0)
                if k in data_dict:
                    traj_pred = data_dict[k]
                    traj_pred = torch.cat([obs[:, None].expand(-1, traj_pred.shape[1], -1, -1), traj_pred], dim=2)
                    all_traj = torch.cat([all_traj, traj_pred], dim=1)
            
            min_pos = torch.amin(all_traj.nan_to_num(all_traj.nanmean()), dim=(0, 1, 2)).cpu().numpy()
            max_pos = torch.amax(all_traj.nan_to_num(all_traj.nanmean()), dim=(0, 1, 2)).cpu().numpy()
        
        diff = max_pos - min_pos

        threshold = 5.0
        center = (max_pos + min_pos) / 2

        max_pos = np.where(diff > threshold, 
                        max_pos + 0.10 * diff, 
                        center + threshold / 2)

        min_pos = np.where(diff > threshold, 
                        min_pos - 0.10 * diff, 
                        center - threshold / 2)
        
        return min_pos, max_pos
                    
    def griddata_on_cluster(self, i, prob, xx, yy, max_pos, min_pos, j, absent_as_nan=False):
        """Rasterise the particle density of step j onto the grid.

        absent_as_nan=False (metrics) keeps the RBF-smoothed values in cells
        with no nearby particle, which is how every Sinkhorn value in the paper
        was computed (code up to c611c33). True (figures only) leaves those
        cells NaN so they are not painted; ot.sinkhorn2 cannot take NaN, which
        made the metric degenerate after e559dc7 introduced it unconditionally.
        """
        prob_ = prob[i, :, j]
        prob_ = prob_[np.where(np.isinf(prob_).sum(axis=1) == 0)]
        prob_ = prob_[np.where(np.isnan(prob_).sum(axis=1) == 0)]
        
        # fill_value stays finite because RBFInterpolator below cannot take NaN; the
        # filled cells are removed again after the smoothing.
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
            smoothing=10, kernel='linear', neighbors=8)
        
        zz = intp(np.array([xx, yy]).reshape(2, -1).T).reshape(xx.shape)

        # zz holds a LOG-density, so the 0 written by griddata's fill_value and by the
        # mask above means density 1, not absence of density -- and the RBF smoothing
        # then spreads that value back over the cells the mask had just cleared. Re-apply
        # the mask afterwards and mark absent cells NaN so they are left unpainted rather
        # than rendered as a moderate density.
        if absent_as_nan:
            zz = np.where(mask, zz, np.nan)
        
        # The channel is a density, not a log-density: metrics.denormalize exponentiates
        # it (traj_metrics.py:86) before the visualizer sees it. So a_min=0 is the natural
        # floor for a non-negative quantity, and a_max caps the interpolation's overshoot
        # at a robust upper bound taken from the particles themselves.
        prob_ = prob[i, :, j, -1]
        prob_ = prob_[~np.isnan(prob_)]
        zz = np.clip(zz, a_min=0.0, a_max=np.percentile(prob_, 95))   # NaN survives

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

    




