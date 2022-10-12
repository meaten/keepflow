from yacs.config import CfgNode
from typing import Dict, List
from pathlib import Path
from abc import ABC, abstractmethod
from joblib import delayed, Parallel
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.density_plot import plot_density
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import griddata


class Visualizer(ABC):
    def __init__(self, cfg: CfgNode):
        pass

    @abstractmethod
    def __call__(self, dict_list: List[Dict]) -> None:
        pass

    def to_numpy(self, tensor) -> np.ndarray:
        return tensor.cpu().numpy()


class TP_Visualizer(Visualizer):
    def __init__(self, cfg: CfgNode):
        self.output_dir = Path(cfg.OUTPUT_DIR) / "visualize"
        self.output_dir.mkdir(exist_ok=True)

    def __call__(self, dict_list: List[Dict]) -> None:
        index = self.to_numpy(dict_list[0]['index'])
        
        # (timesteps, batch, [x,y])
        obs = self.to_numpy(dict_list[0]['obs'])
        gt = self.to_numpy(dict_list[0]['gt'])
        seq_start_end = self.to_numpy(dict_list[0]['seq_start_end'])

        pred = []
        for d in dict_list:
            pred.append(self.to_numpy(d[("pred", 0)][:, :, None]))
            assert np.all(obs == self.to_numpy(d["obs"]))
            assert np.all(gt == self.to_numpy(d["gt"]))

        # (timesteps, batch, num_trials, [x,y])
        pred = np.concatenate(pred, axis=2)
        for i, (s, e) in enumerate(seq_start_end):
            self.plot2d_trajectories(obs[:, s:e],
                                     gt[:, s:e],
                                     pred[:, s:e],
                                     img_path=self.output_dir / f"{index[i]}.png")
        
        
        if ("prob", 0) in dict_list[0]:
            max_pos = self.to_numpy(dict_list[0]["max"])
            min_pos = self.to_numpy(dict_list[0]["min"])
            max_pos += 0.05 * (max_pos - min_pos)
            min_pos -= 0.05 * (max_pos - min_pos)

            num_grid = 1000
            xs = np.linspace(min_pos[0], max_pos[0], num=num_grid)
            ys = np.linspace(min_pos[1], max_pos[1], num=num_grid)
            xx, yy = np.meshgrid(xs, ys)
            
            path_density_map = self.output_dir / "density_map"
            path_density_map.mkdir(exist_ok=True)
            
            for k in dict_list[0].keys():
                if k[0] == "prob":
                    update_step = k[1]
                    prob = self.to_numpy(dict_list[0][k])
                    for i, (s, e) in enumerate(seq_start_end):
                        zz_list = Parallel(n_jobs=len(prob))(delayed(self.griddata_on_cluster)(i, s, e, prob, xx, yy,
                                                                                            max_pos, min_pos, path_density_map,
                                                                                            index, update_step, j)
                                                            for j in range(len(prob)))
                        zz_sum = sum(zz_list)
                        plot_density(xx, yy, zz_sum,
                                        path=path_density_map / f"update{update_step}_{index[i]}_sum.png")
                        
    def griddata_on_cluster(self, i, s, e, prob, xx, yy, max_pos, min_pos, path_density_map, index, update_step, j):
        zz_observed = []
        for k in range(s, e):
            lnk = linkage(prob[j, :, k, :-1],
                            method='single',
                            metric='euclidean')
            idx_cls = fcluster(lnk, t=np.linalg.norm(max_pos-min_pos)*0.001, 
                                criterion='distance')
            idx_cls -= 1
            zz_ = sum([griddata(prob[j, idx_cls == c, k, :-1],
                                prob[j, idx_cls == c, k, -1],
                                (xx, yy), method='linear',
                                fill_value=0.0)
                        for c in range(np.max(idx_cls)+1) if sum(idx_cls == c) > 3]) # at least 4 points needed
            zz_observed.append(zz_)

        zz = sum(zz_observed)
        plot_density(
            xx, yy, zz, path=path_density_map / f"update{update_step}_{index[i]}_{j}.png")
        return zz

    def plot2d_trajectories(self,
                            obs:  np.ndarray,
                            gt:   np.ndarray,
                            pred: np.ndarray,
                            img_path: Path) -> None:
        """plot 2d trajectories

        Args:
            obs (np.ndarray): (num_timesteps, [x, y])
            gt (np.ndarray): (num_timesteps, [x,y])
            pred (np.ndarray): (num_timesteps, num_trials, [x,y])
            img_path (Path): Path
        """

        num_timesteps, num_seqs, num_trials, num_dim = pred.shape
        gt_vis = np.zeros([num_timesteps+1, num_seqs, num_dim])
        gt_vis[0] = obs[-1]
        gt_vis[1:] = gt

        pred_vis = np.zeros([num_timesteps+1, num_seqs, num_trials, num_dim])
        # (num_seqs, num_dim) -> (num_seqs, 1, num_dim)
        pred_vis[0] = obs[-1][:, None]
        pred_vis[1:] = pred

        f, ax = plt.subplots(1, 1)

        for j in range(num_seqs):
            sns.lineplot(x=obs[:, j, 0], y=obs[:, j, 1], color='black',
                         legend='brief', label="obs", marker='o')
            sns.lineplot(x=gt_vis[:, j, 0], y=gt_vis[:, j, 1],
                         color='blue', legend='brief', label="GT", marker='o')
            for i in range(pred.shape[2]):
                if i == 0:
                    sns.lineplot(x=pred_vis[:, j, i, 0], y=pred_vis[:, j, i, 1],
                                 color='green', legend='brief', label="pred", marker='o')
                else:
                    sns.lineplot(
                        x=pred_vis[:, j, i, 0], y=pred_vis[:, j, i, 1], color='green', marker='o')
        plt.savefig(img_path)
        plt.close()
