from typing import List, Dict
import numpy as np
import kornia
import torch
import lpips as LPIPS

lpips_fn = LPIPS.LPIPS(net='alex')

def VP_metrics(dict_list: List) -> Dict:
    
    lpips_fn.cuda()
    
    mse, ssim, psnr, lpips = [], [], [], []
    for data_dict in dict_list:
        gt = data_dict["gt"]
        pred = data_dict[("pred", 0)]
        mse_, ssim_, psnr_, lpips_ = eval_seq(gt, pred)
        mse.append(mse_.cpu().numpy())
        ssim.append(ssim_.cpu().numpy())
        psnr.append(psnr_.cpu().numpy())
        lpips.append(lpips_.cpu().numpy())

    mse = np.array(mse)    
    ssim = np.array(ssim)
    psnr = np.array(psnr)
    lpips = np.array(lpips)
    
    # pick the best sample
    ssim_tmean = ssim.mean(axis=1)
    idxs = ssim_tmean.argmax(axis=0)
    mse = np.array([mse[idx, :, i] for i, idx in enumerate(idxs)]).transpose()
    ssim = np.array([ssim[idx, :, i] for i, idx in enumerate(idxs)]).transpose()
    psnr = np.array([psnr[idx, :, i] for i, idx in enumerate(idxs)]).transpose()
    lpips = np.array([lpips[idx, :, i] for i, idx in enumerate(idxs)]).transpose()
    
    n_sample = mse.shape[1]
    
    mse = np.sum(mse, axis=1)
    ssim = np.sum(ssim, axis=1)
    psnr = np.sum(psnr, axis=1)
    lpips = np.sum(lpips, axis=1)
    
    if np.sum(np.isinf(psnr)):
        import pdb;pdb.set_trace()
    return {"score": np.sum(mse),
            "mse": mse,
            "mean_mse": np.mean(mse),
            "ssim": ssim,
            "mean_ssim": np.mean(ssim),
            "psnr": psnr,
            "mean_psnr": np.mean(psnr),
            "lpips": lpips,
            "mean_lpips": np.mean(lpips),
            "nsample": n_sample}
        

def eval_seq(gt, pred):
    # assume RGB input
    gt = gt[:, :, :3]
    pred = pred[:, :, :3]
    
    T, bs, c, h, w = gt.shape
    gt = gt.reshape(T * bs, c, h, w)
    pred = pred.reshape(T * bs, c, h, w)

    SSIM = kornia.metrics.SSIM(window_size=7, max_val=1.0, padding='valid')  # default of skimage ssim)
    
    psnr = psnr_loss(gt, pred, max_val=1.0, reduction='none')
    ssim = SSIM(gt, pred)
    ssim = torch.mean(ssim, axis=[1,2,3])
    mse = torch.mean((gt - pred) ** 2, axis=[1, 2, 3])
    lpips = lpips_fn(gt, pred)
    
    psnr = psnr.reshape(T, bs)
    ssim = ssim.reshape(T, bs)
    mse = mse.reshape(T, bs)
    lpips = lpips.reshape(T, bs)
    return mse, ssim, psnr, lpips


def psnr_loss(gt, pred, max_val=1.0, reduction='mean'):
    epsilon = 1e-5
    mse = torch.mean((gt - pred) ** 2, axis=[1,2,3])
    psnr = 20 * torch.log10(max_val / (torch.sqrt(mse) + epsilon))
    if reduction == 'mean':
        return torch.mean(psnr)
    elif reduction == 'sum':
        return torch.sum(psnr)
    elif reduction == 'none':
        return psnr
    else:
        raise(ValueError)



