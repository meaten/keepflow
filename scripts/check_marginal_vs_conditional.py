"""Is `analytic_log_prob` the per-step marginal or a one-step conditional?

`FlowChain.log_prob_separate` centres each step's base distribution on the
*previous ground-truth position*, which makes it a teacher-forced conditional
log p(x_{t+n} | x_{t+n-1} = GT). Eqs. (7) to (9) of the manuscript, and the
particle path used for the density grids, instead chain a single base sample
from the latest observed position, which gives the marginal log p(x_{t+n} | o_t).

This script computes both on the same scenes and prints the gap. If they agree,
the two descriptions coincide for this architecture and nothing is wrong. If
they do not, the analytic path is not measuring the quantity the manuscript
claims, and the reported log-probabilities are not comparable with the
baselines' marginals.
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
    p.add_argument("--n_scenes", type=int, default=5)
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    cond_all, marg_all = [], []
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

            conditional = flow.log_prob(base_pos, gt, cond)   # (B, pred_len)

            # Marginal: take each ground-truth position back to the base space
            # through the whole chain, then score it under the single base
            # Gaussian centred on the latest observed position.
            T = flow.pred_len
            marginal = []
            if hasattr(flow, "cif"):
                u, log_r_u = flow.cif.r_u_given_x.sample(
                    torch.cat([gt[:, -1], cond], dim=-1))
                ctx = u
            else:
                ctx = cond
            for n in range(T):
                z, ldj = flow.net(gt[:, n], ctx, steps=list(range(n + 1, -1, -1)))
                z0 = z[:, -1]
                lp = flow.base_dist(base_pos, step=0).log_prob(z0).sum(dim=-1)
                step_lp = lp - ldj[:, -1]
                if hasattr(flow, "cif"):
                    # forward_separate spreads the CIF correction evenly over the
                    # chain, so reproduce the same per-step share here; this makes
                    # the two quantities identical at n = 0 by construction.
                    log_q_u = flow.cif.q_u_given_z.log_prob(
                        u, torch.cat([z0, cond], dim=-1))
                    step_lp = step_lp + (log_q_u - log_r_u) / T
                marginal.append(step_lp)
            marginal = torch.stack(marginal, dim=1)

            cond_all.append(conditional.reshape(-1).cpu().numpy())
            marg_all.append(marginal.reshape(-1).cpu().numpy())
            if i == 0:
                np.set_printoptions(precision=2, suppress=True, linewidth=200)
                print("conditional (analytic_log_prob):", cond_all[-1])
                print("marginal    (chain from base)  :", marg_all[-1])

    C = np.concatenate(cond_all)
    M = np.concatenate(marg_all)
    print(f"\ncells {C.size}")
    print(f"conditional mean {C.mean():9.3f}")
    print(f"marginal    mean {M.mean():9.3f}")
    print(f"difference       {(C - M).mean():9.3f} "
          f"(median {np.median(C - M):.3f}, p10 {np.percentile(C - M, 10):.3f}, "
          f"p90 {np.percentile(C - M, 90):.3f})")


if __name__ == "__main__":
    main()
