"""Are the reported densities in position units for every method?

`TrajectoryMetrics.denormalize` integrates predicted *velocities* into positions
for `PRED_STATE: state_v` models (`x_n = x_prev + dt * cumsum(v)`) but leaves the
density channel untouched apart from `exp()`. If a model's density is produced in
velocity space, the coordinates then live in metres while the density lives in
(m/s)^-2 -- a change-of-variables factor of dt^2 (0.16 for dt=0.4, i.e. 1.83
nats) is missing.

This matters only for models whose density is *analytic and native*: the
KDE-based baselines have their kernels fitted after `denormalize`, so their
densities are in position units by construction.

The test is direct: rasterise the predicted density over a grid and integrate.
A properly normalised position-space density integrates to ~1; a velocity-space
density mistakenly plotted over positions integrates to ~dt^2.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keepflow.utils import load_config, kde
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics
from keepflow.visualization import build_visualizer
from train import dict_to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_scenes", type=int, default=5)
    p.add_argument("--grid_size", type=int, default=400,
                   help="grid resolution for the numerical integral; the default "
                        "100 used for EMD is too coarse to integrate accurately")
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    metrics = build_metrics(cfg)
    visualizer = build_visualizer(cfg)
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    print(f"PRED_STATE = {cfg.DATA.TRAJ.PRED_STATE}   TEST.KDE = {cfg.TEST.KDE}   "
          f"dt-based Jacobian would be dt^2")

    integrals = []
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= args.n_scenes:
                break
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            res = model.predict(data_dict, return_prob=True)
            res = metrics.denormalize(res)
            if cfg.TEST.KDE:
                res = kde(res)
            res = visualizer.prob_on_grid(res, grid_size=args.grid_size)

            zz = res[("prob_grid", 0)]          # (B, G*G, T) density values
            xx, yy = res["grid"]
            xx = np.asarray(xx)
            yy = np.asarray(yy)
            dx = float(np.abs(xx.reshape(-1)[1] - xx.reshape(-1)[0]))
            dy = float(np.abs(np.diff(np.unique(yy.reshape(-1)))[0]))
            z = zz.detach().cpu().numpy() if torch.is_tensor(zz) else np.asarray(zz)
            per_step = z[0].sum(axis=0) * dx * dy   # sum over grid cells, per timestep
            integrals.append(per_step)
            print(f"scene {i}: integral per step = "
                  f"{np.array2string(per_step, precision=3, max_line_width=200)}")

    if integrals:
        a = np.concatenate(integrals)
        print(f"\nmean integral over all scenes/steps: {a.mean():.4f}  "
              f"(median {np.median(a):.4f}, min {a.min():.4f}, max {a.max():.4f})")
        print("~1.00 -> position units;  ~0.16 -> velocity units (missing dt^2)")


if __name__ == "__main__":
    main()
