"""Does the discrete FlowChain accumulate log-det-Jacobians as a sum or an average?

`FlowChain_separate.inverse` divides the cumulative log-det-Jacobian by the step
index before returning it, so the particles carry

    base_log_prob + (1/n) * sum_m ldj_m

while Eq. (9) of the manuscript, and the change-of-variables formula, call for

    base_log_prob + sum_m ldj_m

The update path uses the sum (`torch.cumsum(self.ldjs_tmp[...])`, no division), so
the two halves of the same model already disagree. This script evaluates the
ground truth under both conventions and compares each against the density the
particles actually represent, read off the grid. Whichever matches is the
convention the reported figures have been using.

The CNF chain does not divide, which is why this only affects FlowChain-NF.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keepflow.data import build_dataloader
from keepflow.metrics import build_metrics
from keepflow.models import build_model
from keepflow.utils import load_config, kde
from keepflow.visualization import build_visualizer
from train import dict_to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_scenes", type=int, default=9)
    p.add_argument("--grid_size", type=int, default=100)
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    metrics = build_metrics(cfg)
    visualizer = build_visualizer(cfg)
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    S, A, G, O = [], [], [], []
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= args.n_scenes:
                break
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            node = model.get_node_type(data_dict)
            cond = model._encode(data_dict, node)
            base_pos = model._get_base_pos(data_dict)
            gt = data_dict["gt"].nan_to_num()
            flow = model.flow[node]

            preimages, ldj_cum, _ = flow.chain_preimages(gt, cond)
            T = flow.pred_len
            z = preimages[:, :, 0]                       # base-space preimage
            lp_base = flow.base_dist(base_pos[:, None].expand(-1, T, -1),
                                     step=0).log_prob(z).sum(dim=-1)
            steps = torch.arange(1, T + 1, device=z.device, dtype=z.dtype)
            summed = lp_base - ldj_cum[:, :, 0]
            averaged = lp_base - ldj_cum[:, :, 0] / steps[None]

            one_step = flow.log_prob(base_pos, gt, cond)

            res = model.predict(data_dict, return_prob=True)
            res = metrics.denormalize(res)
            if cfg.TEST.KDE:
                res = kde(res)
            res = visualizer.prob_on_grid(res, grid_size=args.grid_size)
            grid = res[("gt_traj_log_prob", 0)].reshape(-1).cpu().numpy()

            n = min(len(grid), summed.numel())
            S.append(summed.reshape(-1).cpu().numpy()[:n])
            A.append(averaged.reshape(-1).cpu().numpy()[:n])
            O.append(one_step.reshape(-1).cpu().numpy()[:n])
            G.append(grid[:n])

    n_steps = len(S[0])
    Sm, Am, Gm, Om = (np.stack(x).reshape(-1, n_steps) for x in (S, A, G, O))
    print("per-step means (step 1..T)")
    print("  sum    ", np.nanmean(Sm, axis=0).round(2))
    print("  average", np.nanmean(Am, axis=0).round(2))
    print("  one-step", np.nanmean(Om, axis=0).round(2))
    print("  grid   ", np.nanmean(Gm, axis=0).round(2))
    S, A, G, O = map(np.concatenate, (S, A, G, O))
    keep = G > -99.9
    print(f"cells {G.size}, clamped by the grid path {int((~keep).sum())}")
    print(f"{'convention':22s} {'mean':>9s} {'vs grid: mean':>14s} {'median':>9s}")
    for name, X in (("chained, sum", S), ("chained, average", A), ("one-step", O)):
        d = X[keep] - G[keep]
        print(f"{name:22s} {X[keep].mean():9.3f} {d.mean():14.3f} {np.median(d):9.3f}")
    print(f"{'grid (particles)':22s} {G[keep].mean():9.3f}")


if __name__ == "__main__":
    main()
