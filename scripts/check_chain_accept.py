"""Acceptance test for the chained density used by the update rows of Table 3.

Three checks, in order of how much they would tell us if they failed:

1. k = 0 agreement. `log_prob_chain(start_step=0)` and the existing
   `log_prob` describe the same thing at the first step, where the base Gaussian
   is the same object in both. They must agree there.

2. Repeatability. The log-det-Jacobian is a single-sample Hutchinson estimate, so
   the chained value is stochastic. Two identical evaluations bound how much of
   any reported difference is estimator noise rather than signal.

3. Monotone base. With the base Gaussian re-centred on a later observation, the
   density at the steps that remain should not be systematically absurd; we print
   the profile per start step so it can be read rather than asserted.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.utils import load_config
from train import dict_to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_scenes", type=int, default=9)
    p.add_argument("--n_draws", type=int, default=1)
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    one_step, chain0, chain0_rep = [], [], []
    per_start = {k: [] for k in (0, 1, 3, 5)}
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
            base_positions = torch.cat([base_pos[:, None], gt[:, :-1]], dim=1)

            one_step.append(flow.log_prob(base_pos, gt, cond).reshape(-1).cpu().numpy())
            c1 = flow.chain_preimages(gt, cond, n_draws=args.n_draws)
            c2 = flow.chain_preimages(gt, cond, n_draws=args.n_draws)
            chain0.append(flow.log_prob_chain(base_positions, c1, 0).reshape(-1).cpu().numpy())
            chain0_rep.append(flow.log_prob_chain(base_positions, c2, 0).reshape(-1).cpu().numpy())
            for k in per_start:
                v = flow.log_prob_chain(base_positions, c1, k).reshape(-1).cpu().numpy()
                per_start[k].append(v)

    O = np.concatenate(one_step)
    C = np.concatenate(chain0)
    R = np.concatenate(chain0_rep)
    n_steps = len(one_step[0])

    print(f"cells {O.size}  (n_draws={args.n_draws})")
    print(f"[1] first step   one-step {O[::n_steps].mean():8.3f}   "
          f"chain {C[::n_steps].mean():8.3f}   "
          f"diff {np.abs(O[::n_steps] - C[::n_steps]).mean():.3f}")
    print(f"    all steps    one-step {O.mean():8.3f}   chain {C.mean():8.3f}")
    print(f"[2] repeatability  mean |chain - chain'| {np.abs(C - R).mean():8.3f}   "
          f"median {np.median(np.abs(C - R)):.3f}   "
          f"mean of means differ by {abs(C.mean() - R.mean()):.3f}")
    print("[3] mean by start step (NaN = step precedes the update)")
    for k in sorted(per_start):
        v = np.concatenate(per_start[k])
        print(f"    start {k}: mean {np.nanmean(v):8.3f}  over {np.sum(~np.isnan(v))} cells")


if __name__ == "__main__":
    main()
