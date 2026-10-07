"""Acceptance test for the multi-step analytic density.

The density grids, and therefore the reported log-probabilities of the
conference version, come from particles propagated from a single base Gaussian
at the latest observed position. `FlowChain.log_prob_chain` evaluates that same
density analytically at the ground truth, without the grid rasterisation and its
clamp. If the implementation is right the two must agree to interpolation noise,
while the one-step conditional produced by `log_prob` must not.

Run this before trusting any number that `log_prob_chain` produces.
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

    grid_all, chain_all, cond_all = [], [], []
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= args.n_scenes:
                break
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            node = model.get_node_type(data_dict)
            cond = model._encode(data_dict, node)
            base_pos = model._get_base_pos(data_dict)
            gt = data_dict["gt"].nan_to_num().clone()
            flow = model.flow[node]

            base_positions = torch.cat([base_pos[:, None], gt[:, :-1]], dim=1)
            cache = flow.chain_preimages(gt, cond)
            chain = flow.log_prob_chain(base_positions, gt, cond,
                                        start_step=0, cache=cache)
            conditional = flow.log_prob(base_pos, gt, cond)

            res = model.predict(data_dict, return_prob=True)
            res = metrics.denormalize(res)
            if cfg.TEST.KDE:
                res = kde(res)
            res = visualizer.prob_on_grid(res, grid_size=args.grid_size)
            grid = res[("gt_traj_log_prob", 0)].reshape(-1).cpu().numpy()

            n = min(len(grid), chain.numel())
            grid_all.append(grid[:n])
            chain_all.append(chain.reshape(-1).cpu().numpy()[:n])
            cond_all.append(conditional.reshape(-1).cpu().numpy()[:n])
            if i == 0:
                np.set_printoptions(precision=2, suppress=True, linewidth=220)
                print("grid (particles + interpolation):", grid_all[-1])
                print("chain (analytic, multi-step)    :", chain_all[-1])
                print("conditional (analytic, one-step):", cond_all[-1])

    G = np.concatenate(grid_all)
    C = np.concatenate(chain_all)
    D = np.concatenate(cond_all)
    keep = G > -99.9          # drop the cells the grid path clamps
    print(f"\ncells {G.size}, clamped by the grid path {int((~keep).sum())}")
    for name, X in (("chain (multi-step)", C), ("conditional (one-step)", D)):
        d = X[keep] - G[keep]
        print(f"{name:24s} mean {X[keep].mean():8.3f}  "
              f"vs grid: mean diff {d.mean():8.3f}  median {np.median(d):8.3f}  "
              f"p90 {np.percentile(np.abs(d), 90):8.3f}")
    print(f"{'grid':24s} mean {G[keep].mean():8.3f}")


if __name__ == "__main__":
    main()
