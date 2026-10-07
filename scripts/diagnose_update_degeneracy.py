"""Why does the fast update degrade so quickly with the number of updates?

The update reweights the cached particles by a Gaussian centred on the newly
observed position, whose width is the trainable base variance of that step,
`logvar[k]`. That width was fitted as the spread of the base distribution, not as
an observation-noise scale. Meanwhile the particle cloud widens with the horizon.
If the cloud outgrows the kernel, almost all of the weight lands on a handful of
particles and the effective sample size collapses.

This script measures the two quantities side by side: the spread of the cached
particles at step k, and the kernel width sigma_k, together with the resulting
effective sample size and the number of particles holding 90 percent of the
weight.
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
    p.add_argument("--n_scenes", type=int, default=20)
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    rows = {}
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= args.n_scenes:
                break
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            node = model.get_node_type(data_dict)
            flow = model.flow[node]
            res = model.predict(data_dict, return_prob=True)
            particles = res[("prob", 0)][..., :2]      # (1, N, T, 2)
            gt = data_dict["gt"]

            for k in range(1, flow.pred_len):
                cloud = particles[0, :, k - 1]                    # (N, 2)
                sigma = flow.logvar[k].exp()                      # (2,)
                obs = gt[0, k - 1]                                # (2,)
                spread = cloud.std(dim=0)
                dist = (cloud - obs[None]).norm(dim=-1)

                lw = torch.distributions.Normal(obs, sigma).log_prob(cloud).sum(-1)
                w = torch.softmax(lw, dim=0)
                ess = 1.0 / (w ** 2).sum()
                srt, _ = torch.sort(w, descending=True)
                n90 = int((torch.cumsum(srt, 0) < 0.9).sum().item()) + 1

                rows.setdefault(k, []).append([
                    spread.mean().item(), sigma.mean().item(),
                    (spread.mean() / sigma.mean()).item(),
                    dist.min().item(), dist.median().item(),
                    (ess / cloud.shape[0]).item(), n90])

    print(f"{'k':>2} {'cloud sd':>9} {'sigma_k':>8} {'ratio':>7} "
          f"{'min dist':>9} {'med dist':>9} {'ESS/N':>7} {'n(90%)':>7}")
    for k in sorted(rows):
        a = np.array(rows[k]).mean(axis=0)
        print(f"{k:>2} {a[0]:9.3f} {a[1]:8.3f} {a[2]:7.2f} "
              f"{a[3]:9.3f} {a[4]:9.3f} {a[5]:7.3f} {a[6]:7.0f}")


if __name__ == "__main__":
    main()
