import torch
from typing import List, Dict

def TP_metrics(dict_list: List) -> Dict:
    ade, fde = [], []
    for data_dict in dict_list:
        ade.append(displacement_error(data_dict["pred"], data_dict["gt"], mode='raw'))
        fde.append(final_displacement_error(data_dict["pred"][-1], data_dict["gt"][-1], mode='raw'))
    total_traj = data_dict["seq_start_end"][-1][-1]
    ade = evaluate_helper(ade, data_dict["seq_start_end"]) 
    fde = evaluate_helper(fde, data_dict["seq_start_end"]) 
    return {"score": ade.cpu().numpy(), "ade": ade.cpu().numpy(), "fde": fde.cpu().numpy(), "nsample": total_traj.cpu().numpy()}


def evaluate_helper(error, seq_start_end):
    error = torch.stack(error, dim=1)
    min_error_sum = 0
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        min_error_sum += torch.min(_error)
    return min_error_sum


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the euclidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)