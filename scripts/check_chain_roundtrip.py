"""Round-trip test for the backward chain used by `chain_preimages`.

Take a base sample, push it forward through the chain to get a particle and its
log-density, then take that particle back through the chain and evaluate the
density analytically. The two must agree: the backward pass is supposed to be the
exact inverse of the forward one, on the same CIF latent.

This isolates the implementation from the grid rasterisation, so a disagreement
here is a bug in `chain_preimages` and a disagreement only in
`check_chain_logprob.py` is interpolation error.
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
    p.add_argument("--n_scenes", type=int, default=3)
    p.add_argument("--n_particles", type=int, default=16)
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    pos_err, ldj_err, lp_err = [], [], []
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= args.n_scenes:
                break
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            node = model.get_node_type(data_dict)
            cond = model._encode(data_dict, node)
            base_pos = model._get_base_pos(data_dict)
            flow = model.flow[node]
            N = args.n_particles
            T = flow.pred_len

            bp = base_pos[:, None].expand(-1, N, -1)
            cd = cond[:, None].expand(-1, N, -1)
            z0 = flow.base_dist(bp).sample()
            u, log_q_u = flow.cif.q_u_given_z.sample(torch.cat([z0, cd], dim=-1))

            fwd, fwd_ldj_cum = flow.net(z0, u, steps=list(range(0, T + 1)))
            log_r_u = flow.cif.r_u_given_x.log_prob(
                u, torch.cat([fwd[..., -1, :], cd], dim=-1))
            base_lp = flow.base_dist(bp).log_prob(z0).sum(dim=-1)
            denom = torch.cumsum(torch.ones_like(fwd_ldj_cum), dim=2)
            particle_lp = base_lp[..., None] + fwd_ldj_cum \
                + (log_q_u - log_r_u)[..., None] / denom

            for n in range(T):
                y = fwd[..., n, :]
                z_back, ldj_back = flow.net(y, u, steps=list(range(n + 1, -1, -1)))
                pos_err.append((z_back[..., -1, :] - z0).abs().max().item())
                ldj_err.append((ldj_back[..., -1] + fwd_ldj_cum[..., n]).abs().max().item())
                lp = flow.base_dist(bp).log_prob(z_back[..., -1, :]).sum(dim=-1) \
                    - ldj_back[..., -1] + (log_q_u - log_r_u) / (n + 1)
                lp_err.append((lp - particle_lp[..., n]).abs().max().item())

    print(f"max |z_back - z0|            {max(pos_err):.3e}")
    print(f"max |ldj_back + ldj_forward| {max(ldj_err):.3e}")
    print(f"max |logp_back - logp_fwd|   {max(lp_err):.3e}")
    print(f"median |logp_back - logp_fwd| {np.median(lp_err):.3e}")


if __name__ == "__main__":
    main()
