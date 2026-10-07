import numpy as np
import torch
import ot
from yacs.config import CfgNode
from typing import List, Dict

from scipy.interpolate import RectBivariateSpline

from trajdata.dataset_specific.sdd_peds.estimated_homography import SDD_HOMOGRAPHY_SCALES



class TrajMetrics(object):
    def __init__(self, cfg: CfgNode):
        self.device = cfg.DEVICE

        self.pred_len = cfg.DATA.PREDICT_LENGTH
        
        self.scale = cfg.DATA.TRAJ.SCALE
        self.state = cfg.DATA.TRAJ.STATE
        self.pred_state = cfg.DATA.TRAJ.PRED_STATE
            
        self.dataset = cfg.DATA.DATASET_NAME
        
    # TODO : calculate metrics on updated trajectory e.g. ("pred", 1), ("pred", 2)
    def __call__(self, result_dict: Dict) -> Dict:
        ade = displacement_error(result_dict[('pred', 0)][:, :, :self.pred_len], result_dict['gt'][:, None, :self.pred_len])
        fde = final_displacement_error(result_dict[('pred', 0)][:, :, -1], result_dict['gt'][:, None, -1])
        emd = self.emd(result_dict)
        # same prediction against the exact fork/branch posterior (sim_gt.py)
        emd_exact = self.emd(result_dict, gt_key="gt_prob_grid_exact")
        log_prob_mean = self.log_prob_mean_per_update(result_dict)
        log_prob_each = self.log_prob_each_first_step(result_dict)
        log_prob_each_update_6_ahead = self.log_prob_each_update_6_ahead(result_dict)
        ade = evaluate_helper(ade) 
        fde = evaluate_helper(fde)

        ade /= self.scale
        fde /= self.scale
        if self.dataset == 'eth':
            ade /= 0.6
            fde /= 0.6
        if self.dataset == 'sdd':
            scale = [SDD_HOMOGRAPHY_SCALES[result_dict['index'][i][0]]['scale'] for i in range(len(result_dict['index']))]
            scale = torch.Tensor(scale).to(self.device)
            ade /= scale
            fde /= scale
        
        results_dict = {"score": ade.cpu().numpy() + fde.cpu().numpy(),
                        "ade": ade.cpu().numpy(),
                        "fde": fde.cpu().numpy()}
        if log_prob_mean is not None:
            results_dict["log_prob_mean"] = log_prob_mean.cpu().numpy()
        
        if log_prob_each is not None:
            results_dict["log_prob_each_from_6"] = log_prob_each[:, 5:].cpu().numpy()
            results_dict["log_prob_each_from_6_mean"] = log_prob_each[:, 5:].mean(dim=-1).cpu().numpy()

        if log_prob_each_update_6_ahead is not None:
            if len(log_prob_each_update_6_ahead[0]) > 6:
                results_dict["log_prob_each_update_6_ahead"] = log_prob_each_update_6_ahead.cpu().numpy()
                results_dict["log_prob_each_update_6_ahead_mean"] = log_prob_each_update_6_ahead.mean(dim=-1).cpu().numpy()

        if emd is not None:
            results_dict["emd"] = emd.cpu().numpy()
        if emd_exact is not None:
            results_dict["emd_exact"] = emd_exact.cpu().numpy()
        
        return results_dict
    
    def denormalize(self, result_dict: Dict) -> Dict:
        return self.output_to_trajectories(result_dict)
    
    def output_to_trajectories(self, data_dict: Dict) -> Dict:
        if self.pred_state == 'state_p':
            assert 'state_p' in self.state
            data_dict['obs'] = data_dict['obs'][..., 0:2] + data_dict['standardize_offset'][..., 0:2]
            data_dict['gt'] += data_dict['standardize_offset'][..., 0:2]
            if len(data_dict['neighbors_gt']) != 0:  # if neighbors exist
                data_dict['neighbors'] = data_dict['neighbors'][..., 0:2] + data_dict['standardize_offset'][:, None, :, 0:2]
                data_dict['neighbors_gt'] += data_dict['standardize_offset'][:, None, :, 0:2]
            
            for k in data_dict.keys():
                if k[0] == "pred" and type(data_dict[k]) == torch.Tensor:
                    data_dict[k] += data_dict['standardize_offset'][:, None, :, 0:2]
                    
                if k[0] == "prob" and type(data_dict[k]) == torch.Tensor:
                    offset = data_dict['standardize_offset'][:, None, :, 0:2]
                    data_dict[k][..., :2] += offset
                    data_dict[k][..., -1] = data_dict[k][..., -1].exp()
                    
        if self.pred_state == 'state_v':
            assert 'state_p' in self.state
            vel_norm = data_dict['standardize_offset'][..., 2:4].norm(dim=-1)  # assume standardize_data=True
            data_dict['obs'] = data_dict['obs'][..., 0:2] + data_dict['standardize_offset'][..., 0:2]
            data_dict['gt'][..., 0] += vel_norm
            data_dict['gt'] = self.integrate(data_dict['gt'], data_dict['dt'], dim=1) + data_dict['obs'][..., [-1], :]
            if len(data_dict['neighbors_gt']) != 0:  # if neighbors exist
                data_dict['neighbors'] = data_dict['neighbors'][..., 0:2] + data_dict['standardize_offset'][:, None, :, 0:2]
                data_dict['neighbors_gt'][..., 0] += vel_norm[:, None]
                data_dict['neighbors_gt'] = self.integrate(data_dict['neighbors_gt'], data_dict['dt'], dim=2)
                data_dict['neighbors_gt'] += data_dict['neighbors'][..., [-1], :]
            
            for k in data_dict.keys():
                if k[0] == "pred" and type(data_dict[k]) == torch.Tensor:
                    offset = data_dict['obs'][..., [-1], :] if k[1] == 0 else data_dict['gt'][..., [k[1]-1], :]
                    data_dict[k][..., 0] += vel_norm[:, None]
                    data_dict[k] = self.integrate(data_dict[k], data_dict['dt'], dim=2) + offset[:, None]
                if k[0] == "prob" and type(data_dict[k]) == torch.Tensor:
                    offset = data_dict['obs'][:, None, [-1]] if k[1] == 0 else data_dict['gt'][:, None, [k[1]-1]]
                    data_dict[k][..., 0] += vel_norm[:, None]
                    data_dict[k][..., :2] = self.integrate(data_dict[k][..., :2], data_dict['dt'], dim=2) + offset
                    data_dict[k][..., -1] = data_dict[k][..., -1].exp()
        return data_dict
    
    def integrate(self, seq, dt, dim=1):
        return torch.cumsum(seq, dim=dim) * dt.view(dt.shape + (1,) * (dim + 1))
            
    def emd(self, data_dict, gt_key="gt_prob_grid"):
        if ("prob_grid", 0) not in data_dict or (gt_key, 0) not in data_dict:
            return None

        emd_per_step = []
        for k in data_dict.keys():
            if k[0] == gt_key:
                prob = torch.Tensor(data_dict[("prob_grid", k[1])]).to(self.device) + 1e-12
                prob /= prob.sum(dim=1, keepdim=True)
                gt_prob = data_dict[(gt_key, k[1])].to(self.device).to(torch.float32)
                gt_prob /= gt_prob.sum(dim=1, keepdim=True)
                """
                target_frame = 7
                gt_prob = torch.Tensor(data_dict["gt_prob"])[:, :, target_frame:]
                gt_prob /= gt_prob.sum(dim=1, keepdim=True)
                prob = np.zeros_like(data_dict[("prob", 0)][..., target_frame:])
                for k in data_dict.keys():
                    if k[0] == "prob":
                        prob[..., k[1]:] = data_dict[k][..., target_frame:]
                prob = torch.Tensor(prob).to(self.device) + 1e-8
                """
                X, Y = data_dict["grid"]
                coords = torch.Tensor(np.array([X.flatten(), Y.flatten()]).T).to(self.device)
                coordsSqr = torch.sum(coords**2, dim=1)
                M = coordsSqr[:, None] + coordsSqr[None, :] - 2*coords.matmul(coords.T)
                M[M < 0] = 0
                M = torch.sqrt(M)
                emd = []
                for b in range(prob.shape[0]):
                    emd_ = []
                    for t in range(prob.shape[-1]):
                        # emd__ = ot.sinkhorn2(gt_prob[0, :, t], prob[0, :, t], M, 1.0 / M.median(),
                        #                       numItermax=300, stopThr=1e-4, warn=False, method='sinkhorn_log')
                        emd__ = ot.sinkhorn2(prob[0, :, t], gt_prob[0, :, t], M, 1.0, warn=False)
                        # print(emd__)
                        emd_.append(emd__)
                    emd.append(torch.Tensor(emd_).mean(dim=-1))
                emd = torch.Tensor(emd)
                # print(emd)
                emd_per_step.append(emd)
        emd_per_step = torch.stack(emd_per_step, dim=1)
        
        return emd_per_step
                
    def log_prob_mean_per_update(self, data_dict):
        # we assume batch_size = 1
        if ("gt_traj_log_prob", 0) in data_dict:
            relevant_keys = [k for k in data_dict.keys() if k[0] == "gt_traj_log_prob"]
            relevant_keys.sort(key=lambda x: x[1])
            
            log_prob = []
            for k in relevant_keys:
                log_prob.append(data_dict[k].nanmean(dim=-1))
            
            log_prob = torch.stack(log_prob, dim=1)
            return log_prob
        else:
            return None
        
    def log_prob_each_first_step(self, data_dict):
        k = ("gt_traj_log_prob", 0) 
        if k in data_dict:
            return data_dict[k]
        else:
            return None
        
    def log_prob_each_update_6_ahead(self, data_dict):
        if ("gt_traj_log_prob", 0) in data_dict:
            relevant_keys = [k for k in data_dict.keys() if k[0] == "gt_traj_log_prob"]
            relevant_keys.sort(key=lambda x: x[1])
            
            log_prob = []
            for k in relevant_keys:
                log_prob.append(data_dict[k][:, 5])
            
            log_prob = torch.stack(log_prob, dim=1)
            return log_prob
        else:
            return None
        
    def compute_road_violations(self, node_type, data_dict):
        channel = 0
        
        index = data_dict["index"]
        predicted_trajs = data_dict[("pred", 0)][:, -self.predict_length:].cpu().numpy()
        try:
            idx_scene = [s.name for s in self.env.scenes]
            idx = [idx_scene.index(i[0])  for i in index]
            scenes = [self.env.scenes[i] for i in idx]
            maps = [s.map[node_type] for s in scenes]
        except:  # for dataset without maps
            return torch.Tensor([0.0])
        maps_data = [1 - m.data[..., channel, :, :] / 255 for m in maps]
        interp_obs_maps = [RectBivariateSpline(range(m.shape[0]), range(m.shape[1]), m, kx=1, ky=1) for m in maps_data]
        
        viol_trajs = []
        for i in range(len(interp_obs_maps)):
            old_shape = predicted_trajs[i].shape
            pred_trajs_map = maps[i].to_map_points(predicted_trajs[i].reshape((-1, 2)))
            traj_obs_values = interp_obs_maps[i](pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
            #traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1], old_shape[2]))
            #num_viol_trajs = np.sum(traj_obs_values.max(axis=2) > 0, dtype=float)
            #traj_obs_values = traj_obs_values.reshape((old_shape[0]))
            viol_trajs.append(np.mean(traj_obs_values.max() > 0, dtype=float))
        
        return torch.tensor(viol_trajs)
    
    def compute_col(self, node_type, data_dict):
        thres=0.2
        num_interp=4
        predicted_traj = data_dict[("pred", 0)][:, -self.predict_length:].cpu()
        neighbors = restore(data_dict["neighbors_gt_pos"])
        edge_types = [edge_type for edge_type in self.env.get_edge_types() if edge_type[0] == node_type]
        col = []
        for i, traj in enumerate(predicted_traj):
            neighbors_cur = []
            for edge_type in edge_types:
                neighbors_cur += neighbors[edge_type][i]
            if len(neighbors_cur) == 0:  # no neighbor
                col.append(False)
                continue
            neighbors_cur = np.array([np.array(neighbor) for neighbor in neighbors_cur])
            dense_all = interpolate_traj(neighbors_cur, num_interp)
            dense_ego = interpolate_traj(traj[None, :], num_interp)
            distances = np.linalg.norm(dense_all - dense_ego, axis=-1)
            mask = distances[:, 0] > 0
            col.append(np.any(distances[mask].min(axis=0) < thres))
        return torch.tensor(col)
    
def interpolate_traj(traj, num_interp=4):
    '''
    Add linearly interpolated points of a trajectory
    '''
    sz = traj.shape
    dense = np.zeros((sz[0], (sz[1] - 1) * (num_interp + 1) + 1, 2))
    dense[:, :1, :] = traj[:, :1]

    for i in range(num_interp+1):
        ratio = (i + 1) / (num_interp + 1)
        dense[:, i+1::num_interp+1, :] = traj[:, 0:-1] * (1 - ratio) + traj[:, 1:] * ratio

    return dense
        

def evaluate_helper(error):
    min_error = torch.min(error, dim=1)[0]
    return min_error

def displacement_error(pred_traj, gt_traj):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - gt_traj: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    Output:
    - loss: gives the eculidian displacement error
    """    
    return torch.norm(pred_traj - gt_traj, dim=-1).nanmean(dim=-1)
    

def final_displacement_error(pred_pos, gt_pos):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - gt_pos: Tensor of shape (batch, 2). Groud truth
    last pos
    Output:
    - loss: gives the euclidian displacement error
    """
    return torch.norm(pred_pos - gt_pos, dim=-1)


def retrieve_scale_sdd(data_dict):
    scene_names = [triplet[0] for triplet in data_dict['index']]
    scales = np.array([SDD_HOMOGRAPHY_SCALES[sn]['scale'] for sn in scene_names])
    return torch.Tensor(scales)
    
    