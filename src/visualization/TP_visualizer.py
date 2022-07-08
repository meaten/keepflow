from yacs.config import CfgNode
from typing import Dict, List
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.density_plot import plot_density


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
            pred.append(self.to_numpy(d["pred"][:, :, None]))
            assert np.all(obs == self.to_numpy(d["obs"]))
            assert np.all(gt == self.to_numpy(d["gt"]))
            
        # (timesteps, batch, num_trials, [x,y])
        pred = np.concatenate(pred, axis=2)
        
        
        for i, (s, e) in enumerate(seq_start_end):
            if s - e == 1:
                import pdb;pdb.set_trace()
            self.plot2d_trajectories(obs[:, s:e],
                                     gt[:, s:e],
                                     pred[:, s:e],
                                     img_path=self.output_dir / f"{index[i]}.png")
            
            
        if "prob" in dict_list[0]:
            prob = self.to_numpy(dict_list[0]["prob"])
            shape = prob.shape
            grid_shape = [shape[0], shape[1], int(np.sqrt(shape[2])), int(np.sqrt(shape[2])), shape[3]]
            prob = np.reshape(prob, grid_shape)
            path_density_map = self.output_dir / "density_map"
            path_density_map.mkdir(exist_ok=True)
            for i, (s, e) in enumerate(seq_start_end):
                prob_seqs = np.sum(prob[:, s:e, :, :, 2], axis=1)
                for j in range(len(gt)):
                    plot_density(prob[j, i, :, :, 0],
                                 prob[j, i, :, :, 1],
                                 prob_seqs[j, :, :],
                                 path=path_density_map / f"{index[i]}_{j}.png")
                    
                prob_sum = np.sum(prob_seqs, axis=0)
                prob_sum = np.clip(prob_sum, 0., 1.)
                plot_density(prob[j, i, :, :, 0],
                             prob[j, i, :, :, 1],
                             prob_sum,
                             path=path_density_map / f"{index[i]}_sum.png")
                        
                
            
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
        pred_vis[0] = obs[-1][:, None]  # (num_seqs, num_dim) -> (num_seqs, 1, num_dim)
        pred_vis[1:] = pred
        
        f, ax = plt.subplots(1,1)
        
        for j in range(num_seqs):
            sns.lineplot(x=obs[:, j, 0], y=obs[:, j, 1], color='black', legend='brief', label="obs", marker='o')
            sns.lineplot(x=gt_vis[:, j, 0], y=gt_vis[:, j, 1],  color='blue', legend='brief', label="GT", marker='o')
            for i in range(pred.shape[1]):
                if i == 0:
                    sns.lineplot(x=pred_vis[:, j, i, 0], y=pred_vis[:, j, i, 1], color='green', legend='brief', label="pred", marker='o')
                else:
                    sns.lineplot(x=pred_vis[:, j, i, 0], y=pred_vis[:, j, i, 1], color='green', marker='o')
        plt.savefig(img_path)
        plt.close()