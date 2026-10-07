"""Does FlowChain's *analytic* density integrate to one?

Two separate quantities are easy to conflate:

  (a) the model's analytic density, evaluated by inverting the flow and adding
      the log-det-jacobians -- this is what the paper claims as its advantage;
  (b) the number actually reported as `log_prob`, which for FlowChain is a
      `LinearNDInterpolator` lookup into a 100x100 rasterisation of (a) built by
      `TrajectoryVisualizer.prob_on_grid`, clamped to -100 outside the convex
      hull (`traj_visualizer.py:164-178`).

Rasterising (a) over a grid and integrating gave ~5.9 for FlowChain-CNF and
~2.5 for FlowChain-NF, against ~1.0 for Trajectron++ and Social-GAN. That is
either a rasterisation artefact or a genuine normalisation error in (a); only
the second would invalidate the reported log-probabilities.

This script settles it by evaluating the analytic log-density directly on a
dense grid -- no particles, no interpolation -- and integrating.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keepflow.utils import load_config
from keepflow.data import build_dataloader
from keepflow.models import build_model
from train import dict_to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_scenes", type=int, default=3)
    p.add_argument("--grid_size", type=int, default=200)
    p.add_argument("--half_width", type=float, default=12.0,
                   help="half-width (in the model's own prediction space) of the "
                        "square integration domain, centred on the predicted mean")
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    print(f"PRED_STATE = {cfg.DATA.TRAJ.PRED_STATE}  (the integral is taken in "
          f"whatever space the model predicts in, so it should be 1 either way)")

    node = None
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= args.n_scenes:
                break
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            node = model.get_node_type(data_dict)
            cond = model._encode(data_dict, node)
            base_pos = model._get_base_pos(data_dict)

            # Anchor the integration domain on the model's own samples so the
            # square actually contains the mass.
            res = model.predict(data_dict, return_prob=True)
            pts = res[("prob", 0)][0, :, :, :2]            # (N, T, 2), raw space
            centre = pts.mean(dim=0)                       # (T, 2)

            T = centre.shape[0]
            g = args.grid_size
            lin = torch.linspace(-args.half_width, args.half_width, g, device=cfg.DEVICE)
            gx, gy = torch.meshgrid(lin, lin, indexing="ij")
            cell = float((lin[1] - lin[0]) ** 2)

            integrals = []
            for t in range(T):
                q = torch.stack([gx.reshape(-1) + centre[t, 0],
                                 gy.reshape(-1) + centre[t, 1]], dim=-1)  # (g*g, 2)
                # one full trajectory per grid point, with only step t varying:
                # the flow's log_prob is a per-step factorised quantity, so the
                # other steps are filled with the predicted mean and the step-t
                # marginal is read off.
                traj = centre[None].repeat(q.shape[0], 1, 1)
                traj[:, t] = q
                lp = model.flow[node].log_prob(
                    base_pos.repeat(q.shape[0], 1),
                    traj,
                    cond.repeat(q.shape[0], 1))
                lp = lp.reshape(q.shape[0], -1)[:, t] if lp.dim() > 1 else lp
                integrals.append(float(lp.exp().sum() * cell))
            print(f"scene {i}: per-step integral of the analytic density = "
                  f"{np.array2string(np.array(integrals), precision=3, max_line_width=250)}",
                  flush=True)

    print("\n1.00 -> the analytic density is a proper density; "
          "a constant factor c -> log-probabilities are inflated by log(c)")


if __name__ == "__main__":
    main()
