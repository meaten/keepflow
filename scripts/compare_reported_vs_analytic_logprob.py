"""How much does the reported log-probability differ from the analytic one?

For models with no native `gt_traj_log_prob` (the FlowChain variants), the
reported number is produced by `TrajectoryVisualizer.prob_on_grid`: the density
is rasterised onto a 100x100 grid built from the particle cloud, and the value
at the ground truth is obtained with `LinearNDInterpolator`, clamped to -100
outside the convex hull (`traj_visualizer.py:164-178`).

The paper's stated advantage, however, is that the density is *analytic*: the
flow can be evaluated at the ground truth directly. This script measures the gap
between the two, per scene and per step, so that the cost of the grid detour --
and the frequency of the -100 clamp -- is quantified rather than assumed.
"""
import argparse
import json
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
    p.add_argument("--n_scenes", type=int, default=9)
    p.add_argument("--grid_size", type=int, default=100,
                   help="must match the grid used by test.py (100) to reproduce "
                        "the reported numbers")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    metrics = build_metrics(cfg)
    visualizer = build_visualizer(cfg)
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    rep_all, ana_all = [], []
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= args.n_scenes:
                break
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            node = model.get_node_type(data_dict)
            cond = model._encode(data_dict, node)
            base_pos = model._get_base_pos(data_dict)
            gt_raw = data_dict["gt"].clone()

            res = model.predict(data_dict, return_prob=True)
            # analytic: evaluate the flow at the ground truth, in the model's own
            # (pre-denormalise) space, which is where the flow is defined.
            analytic = model.flow[node].log_prob(base_pos, gt_raw.nan_to_num(), cond)
            analytic = analytic.reshape(-1).cpu().numpy()

            res = metrics.denormalize(res)
            if cfg.TEST.KDE:
                res = kde(res)
            res = visualizer.prob_on_grid(res, grid_size=args.grid_size)
            reported = res[("gt_traj_log_prob", 0)].reshape(-1).cpu().numpy()

            n = min(len(reported), len(analytic))
            rep_all.append(reported[:n])
            ana_all.append(analytic[:n])
            print(f"scene {i}: reported={np.array2string(reported[:n], precision=2, max_line_width=250)}",
                  flush=True)
            print(f"{'':9s} analytic={np.array2string(analytic[:n], precision=2, max_line_width=250)}")

    if not rep_all:
        return
    R = np.concatenate(rep_all)
    A = np.concatenate(ana_all)
    d = R - A
    print(f"\ncells: {R.size}")
    print(f"reported mean : {R.mean():10.3f}")
    print(f"analytic mean : {A.mean():10.3f}")
    print(f"difference    : {d.mean():10.3f}  (median {np.median(d):.3f}, "
          f"p10 {np.percentile(d, 10):.3f}, p90 {np.percentile(d, 90):.3f})")
    print(f"cells clamped to -100 in the reported path: {int((R <= -99.9).sum())}/{R.size}")
    if args.out:
        with open(args.out, "w") as fp:
            json.dump({"config": args.config_file,
                       "reported": R.tolist(), "analytic": A.tolist(),
                       "reported_mean": float(R.mean()),
                       "analytic_mean": float(A.mean()),
                       "n_clamped": int((R <= -99.9).sum())}, fp, indent=2)


if __name__ == "__main__":
    main()
