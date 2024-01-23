import torch.nn.functional as F

class CollateWrapper(object):
    def __init__(self, cfg, collate_fn, scale) -> None:
        self.collate_fn = collate_fn
        
        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len = cfg.DATA.PREDICT_LENGTH
        self.state = cfg.DATA.TRAJ.STATE
        self.pred_state = cfg.DATA.TRAJ.PRED_STATE
        
        if self.state == 'state_pva':
            self.state_idx = [0, 1, 2, 3, 4, 5]
        elif self.state == 'state_p':
            self.state_idx = [0, 1]
        elif self.state == 'state_v':
            self.state_idx = [2, 3]
            
        if self.pred_state == 'state_p':
            self.pred_state_idx = [0, 1]
        elif self.pred_state == 'state_v':
            self.pred_state_idx = [2, 3]
            
        self.scale = scale
        
    def __call__(self, batch_):
        batch = self.collate_fn(batch_)
        
        index = [(s, t, f'{tp}_{a}') for s, t, tp, a in zip(batch.scene_ids, batch.scene_ts, batch.agent_type, batch.agent_name)]
        
        curr_pos = batch.curr_agent_state[:, None, :2]

        obs = batch.agent_hist
        obs[..., :2] -= curr_pos
        obs = obs[..., self.state_idx]
        
        gt = batch.agent_fut
        gt[..., :2] -= curr_pos
        gt = gt[..., self.pred_state_idx]

        curr_pos = curr_pos[:, None]

        neighbors = batch.neigh_hist
        neighbors[..., :2] -= curr_pos
        neighbors = neighbors[..., self.state_idx]
        neighbors = F.pad(neighbors, (0, 0, self.obs_len - neighbors.shape[2], 0), 'constant', float('nan'))
        
        neighbors_gt = batch.neigh_fut
        neighbors_gt[..., :2] -= curr_pos
        neighbors_gt = neighbors_gt[..., self.pred_state_idx]
        neighbors_gt = F.pad(neighbors_gt, (0, 0, 0, self.pred_len - neighbors_gt.shape[2]), 'constant', float('nan'))
        
        curr_pos *= self.scale
        obs *= self.scale
        gt *= self.scale
        neighbors *= self.scale
        neighbors_gt *= self.scale
        
        data_dict = {
            "index": index,
            "obs": obs,
            "gt": gt,
            # "obs_st": obs,
            # "gt_st": gt,
            # "neighbors_st": neighbors,
            # "neighbors_gt_st": neighbors_gt,
            "neighbors": neighbors,
            "neighbors_gt": neighbors_gt,
            # "neighbors_edge": neighbors_edge_value,
            'agent_type': batch.agent_type,
            'neighbor_type': batch.neigh_types,
            'curr_state': batch.curr_agent_state,
            "robot_traj": batch.robot_fut,
            "map": batch.maps,
            "dt": batch.dt,
            "first_history_index": batch.agent_hist.shape[1] - batch.agent_hist_len,
            # "this_node_type": batch.agent_type
        }
        
        return data_dict