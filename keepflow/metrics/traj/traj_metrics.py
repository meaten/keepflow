import numpy as np
import torch
import ot
from yacs.config import CfgNode
from typing import List, Dict
# import warnings
# warnings.simplefilter('ignore')


class TrajMetrics(object):
    def __init__(self, cfg: CfgNode):
        self.device = cfg.DEVICE
        
        self.pred_len = cfg.DATA.PREDICT_LENGTH

        self.state = cfg.DATA.TRAJ.STATE
        self.pred_state = cfg.DATA.TRAJ.PRED_STATE
            
        self.dataset = cfg.DATA.DATASET_NAME
        
    # TODO : calculate metrics on updated trajectory e.g. ("pred", 1), ("pred", 2)
    def __call__(self, dict_list: List) -> Dict:
        ade, fde, emd, log_prob = [], [], [], []
        for data_dict in dict_list:
            ade.append(displacement_error(data_dict[('pred', 0)][:, :self.pred_len], data_dict['gt'][:, :self.pred_len]))
            fde.append(final_displacement_error(data_dict[('pred', 0)][:, -1], data_dict['gt'][:, -1]))
            emd.append(self.emd(data_dict))
            log_prob.append(self.log_prob(data_dict))
        
        ade = evaluate_helper(ade) 
        fde = evaluate_helper(fde)
        emd = evaluate_helper(emd)
        log_prob = evaluate_helper(log_prob)

        if self.dataset == 'eth':
            ade /= 0.6
            fde /= 0.6
        # if self.dataset == 'sdd':
        #     ade = ade * 50
        #     fde = fde * 50
        
        return {"score": ade.cpu().numpy(),
                "ade": ade.cpu().numpy(),
                "fde": fde.cpu().numpy(),
                "emd": emd.cpu().numpy(),
                "log_prob": log_prob.cpu().numpy()}
    
    def denormalize(self, dict_list: List) -> List:
        for data_dict in dict_list:
            # if not ("pred", 0) in data_dict.keys():
                # data_dict = self.unstandardize(data_dict)
            data_dict = self.output_to_trajectories(data_dict)
        return dict_list

    def output_to_trajectories(self, data_dict: Dict) -> Dict:
        if self.pred_state == 'state_p':
            assert 'state_p' in self.state
            data_dict['obs'] = data_dict['obs'][..., 0:2] + data_dict['curr_state'][:, None, 0:2]
            data_dict[('pred', 0)] += data_dict['curr_state'][:, None, 0:2]
            data_dict['gt'] += data_dict['curr_state'][:, None, 0:2]
            if len(data_dict['neighbors_gt']) != 0:  # if neighbors exist
                data_dict['neighbors'] = data_dict['neighbors'][..., 0:2] + data_dict['curr_state'][:, None, None, 0:2]
                data_dict['neighbors_gt'] += data_dict['curr_state'][:, None, None, 0:2]
            
            for k in data_dict.keys():
                if k[0] == "prob" and type(data_dict[k]) == torch.Tensor:
                    offset = data_dict['curr_state'][:, None, None, 0:2] if k[1] == 0 else data_dict['gt'][:, None, k[1]-1:k[1], 0:2]
                    data_dict[k][..., :2] += offset
            
        if self.pred_state == 'state_v':
            assert 'state_p' in self.state
            data_dict['obs'] = data_dict['obs'][..., 0:2] + data_dict['curr_state'][:, None, 0:2]
            data_dict[('pred', 0)] = self.integrate(data_dict[('pred', 0)], data_dict['dt'], data_dict['curr_state'][:, 0:2], dim=1)
            data_dict['gt'] = self.integrate(data_dict['gt'], data_dict['dt'], data_dict['curr_state'][:, 0:2], dim=1)
            if len(data_dict['neighbors_gt']) != 0:  # if neighbors exist
                data_dict['neighbors'] = data_dict['neighbors'][..., 0:2] + data_dict['curr_state'][:, None, None, 0:2]
                data_dict['neighbors_gt'] = self.integrate(data_dict['neighbors_gt'], data_dict['dt'], data_dict['neighbors'][:, :, -1], dim=2)

            for k in data_dict.keys():
                if k[0] == "prob" and type(data_dict[k]) == torch.Tensor:
                    offset = data_dict['curr_state'][:, 0:2] if k[1] == 0 else data_dict['gt'][:, k[1]-1, 0:2]
                    data_dict[k][..., :2] = self.integrate(data_dict[k][..., :2], data_dict['dt'], offset, dim=2)
            
        return data_dict
    
    def integrate(self, seq, dt, c, dim=1):
        return torch.cumsum(seq, dim=dim) * dt.view(dt.shape + (1,) * (dim + 1)) + c.view((*c.shape[:-1],) + (1,) + (c.shape[-1],))
    
    def emd(self, data_dict):
        if ("prob", 0) not in data_dict or ("gt_prob") not in data_dict:
            return torch.Tensor([0.0])
        
        key = None
        for k in data_dict.keys():
            if k[0] == "prob":
                key = k
        prob = torch.Tensor(data_dict[key]).to(self.device) + 1e-12
        prob /= prob.sum(dim=1, keepdim=True)
        gt_prob = torch.Tensor(data_dict["gt_prob"])[:, :, key[1]:]
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
                emd__ = ot.sinkhorn2(prob[0, :, t], gt_prob[0, :, t], M, 1.0, warn=False)
                # print(emd__)
                emd_.append(emd__)
            emd.append(torch.Tensor(emd_).mean(dim=-1))
        emd = torch.Tensor(emd)
        # print(emd)
        return emd
                
    def log_prob(self, data_dict):
        # we assume batch_size = 1
        if ("gt_traj_log_prob", 0) in data_dict:        
            return data_dict[("gt_traj_log_prob", 0)].nanmean(dim=-1)
        else:
            return torch.Tensor([0.0])
        

def evaluate_helper(error):
    error = torch.stack(error, dim=0)
    min_error_sum = torch.min(error, dim=0)[0]
    return min_error_sum

def displacement_error(pred_traj, gt_traj):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - gt_traj: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    Output:
    - loss: gives the eculidian displacement error
    """    
    return torch.norm(pred_traj - gt_traj, dim=2).nanmean(dim=1)
    

def final_displacement_error(pred_pos, gt_pos):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - gt_pos: Tensor of shape (batch, 2). Groud truth
    last pos
    Output:
    - loss: gives the euclidian displacement error
    """
    return torch.norm(pred_pos - gt_pos, dim=1)


    
    