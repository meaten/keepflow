from copy import deepcopy
from typing import Dict, List
import numpy as np
import torch
from torch.autograd.variable import Variable


def MP_metrics(dict_list: List) -> Dict:
    error = []
    eval_frame = [1, 3, 7, 9]
    for data_dict in dict_list:
        
        targ_expmap = data_dict["gt_all"]
        pred_expmap = deepcopy(data_dict["gt_all"])
        dim_used = data_dict["dim_used"].cpu().numpy().astype(int)
        
        pred_expmap[:, :, dim_used] = data_dict["pred"][:, :, :54]
        pred_expmap = pred_expmap * data_dict["std"] + data_dict["mean"]
        
        targ_expmap = targ_expmap.transpose(1, 0)
        pred_expmap = pred_expmap.transpose(1, 0)

        bs, output_n, dim_full_len = targ_expmap.shape
        t_e = np.zeros([len(eval_frame), bs])
        
        targ_expmap = targ_expmap.view(-1, dim_full_len)
        pred_expmap = pred_expmap.view(-1, dim_full_len)

        pred_expmap[:, 0:6] = 0
        targ_expmap[:, 0:6] = 0
        pred_expmap = pred_expmap.view(-1, 3)
        targ_expmap = targ_expmap.view(-1, 3)

        # get euler angles from expmap
        pred_eul = rotmat2euler_torch(
            expmap2rotmat_torch(pred_expmap))
        pred_eul = pred_eul.view(-1,
                                 dim_full_len).view(-1,
                                                    output_n,
                                                    dim_full_len)
        targ_eul = rotmat2euler_torch(
            expmap2rotmat_torch(targ_expmap))
        targ_eul = targ_eul.view(-1,
                                 dim_full_len).view(-1,
                                                    output_n,
                                                    dim_full_len)
        # update loss and testing errors
        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_e[k] += torch.linalg.norm(pred_eul[:, j, :] - targ_eul[:, j, :], 2, 1).cpu().data.numpy()
        error.append(deepcopy(t_e))

    error = np.array(error)

    # pick the best sample
    error_tmean = error.mean(axis=1)
    idxs = error_tmean.argmin(axis=0)
    error = np.array([error[idx, :, i] for i, idx in enumerate(idxs)]).transpose()

    return {"score": np.sum(error), "error": np.sum(error, axis=1), "nsample": bs}

def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above

    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = Variable(torch.zeros(n, 3).float()).cuda()
    idx_spec1 = (R[:, 0, 2] == 1).nonzero(as_tuple=False
    ).cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -
                 1).nonzero(as_tuple=False).cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = Variable(torch.zeros(len(idx_spec1), 3).float()).cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = Variable(torch.zeros(len(idx_spec2), 3).float()).cuda()
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(
        idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = Variable(torch.zeros(len(idx_remain), 3).float()).cuda()
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
        eul[idx_remain, :] = eul_remain

    return eul


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = Variable(torch.eye(3, 3).repeat(n, 1, 1)).float().cuda() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R

        