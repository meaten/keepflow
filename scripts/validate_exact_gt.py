"""Validate the exact fork/branch ground truth (keepflow/data/traj/sim_gt.py).

1. Population check. Simulate the generator (noise on all L+T points, frame
   centred on the noisy last observation) and histogram each future step on a
   fixed 100x100 grid. Averaging the exact conditional density over many
   simulated observation windows must reproduce that histogram; the stored GMM
   (std 0.09) should not. Distances are the paper's Sinkhorn (reg 1.0,
   Euclidean cost on grid coordinates, both sides normalised). A second,
   independent histogram gives the sampling-noise floor.
2. Conditioning check. On the real test set, HPD levels of the ground truth
   under the posterior conditioned on the history and on the first k future
   steps (k = 0, 1, 3, 5), per step and joint; calibrated => u ~ U(0,1).

    pyenv exec python scripts/validate_exact_gt.py --dataset simfork --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import dill
import numpy as np
import ot
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from keepflow.utils import load_config
from keepflow.data import build_dataloader
from keepflow.metrics import build_metrics
from keepflow.data.traj.sim_gt import ExactForkBranchGT, FORK_NOISE_STD
from train import dict_to_device

ALPHAS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def simulate(mu, n, L, s, rng):
    """(n, T, 2) future positions relative to the noisy last observation, and
    (n, L, 2) the observation window in that frame."""
    T, K, _ = mu.shape
    line = np.stack([np.arange(-(L - 1), 1.0), np.zeros(L)], 1)
    k = rng.integers(K, size=n)
    obs = line[None] + s * rng.standard_normal((n, L, 2))
    fut = mu[:, k].transpose(1, 0, 2) + s * rng.standard_normal((n, T, 2))
    origin = obs[:, -1:].copy()
    return fut - origin, obs - origin


def hist(points, edges_x, edges_y):
    h, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[edges_x, edges_y])
    return h.T  # rows = y, like meshgrid


def sinkhorn(p, q, M, device):
    p = torch.as_tensor(p.ravel() + 1e-12, dtype=torch.float32, device=device)
    q = torch.as_tensor(q.ravel() + 1e-12, dtype=torch.float32, device=device)
    return ot.sinkhorn2(p / p.sum(), q / q.sum(), M, 1.0, warn=False).item()


def sinkhorn_div(p, q, M, device):
    """Debiased Sinkhorn divergence S(p,q) - S(p,p)/2 - S(q,q)/2 (zero iff p = q)."""
    return sinkhorn(p, q, M, device) - 0.5 * sinkhorn(p, p, M, device) - 0.5 * sinkhorn(q, q, M, device)


def coarse(h, f=2):
    G = h.shape[0]
    return h.reshape(G // f, f, G // f, f).sum((1, 3))


def emd_exact(p, q, M):
    p = p.ravel().astype(np.float64) + 1e-12
    q = q.ravel().astype(np.float64) + 1e-12
    return float(ot.emd2(p / p.sum(), q / q.sum(), M, numItermax=2_000_000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="simfork")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_sim", type=int, default=400000)
    ap.add_argument("--n_obs_avg", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.config_file = f"configs/traj/FlowMatchingChain_onestep/CIF_spline/{args.dataset}.yml"
    cfg = load_config(args)
    dev = cfg.DEVICE
    env = dill.load(open(Path(cfg.DATA.PATH) / cfg.DATA.TASK / "processed_data"
                         / f"{args.dataset}_test.pkl", "rb"), encoding="latin1")
    stored = env.gt_dist
    gt = ExactForkBranchGT(stored, obs_len=cfg.DATA.OBSERVE_LENGTH)
    mu = gt.mu.numpy()
    T = mu.shape[0]
    L, s = cfg.DATA.OBSERVE_LENGTH, FORK_NOISE_STD
    rng = np.random.default_rng(0)

    # ---------------- 1. population check ----------------
    lo = np.array([mu[..., 0].min() - 2.5, mu[..., 1].min() - 2.5])
    hi = np.array([mu[..., 0].max() + 2.5, mu[..., 1].max() + 2.5])
    G = 100
    xs, ys = np.linspace(lo[0], hi[0], G), np.linspace(lo[1], hi[1], G)
    xx, yy = np.meshgrid(xs, ys)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    ex = np.concatenate([xs - dx / 2, [xs[-1] + dx / 2]])
    ey = np.concatenate([ys - dy / 2, [ys[-1] + dy / 2]])
    coords = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float32, device=dev)
    M = torch.cdist(coords, coords)
    # coarse 50x50 grid (cell centres) for the exact EMD
    xc = xx.reshape(G // 2, 2, G // 2, 2).mean((1, 3))
    yc = yy.reshape(G // 2, 2, G // 2, 2).mean((1, 3))
    cc = np.stack([xc.ravel(), yc.ravel()], 1)
    Mc = np.sqrt(((cc[:, None] - cc[None]) ** 2).sum(-1))

    fut_a, _ = simulate(mu, args.n_sim, L, s, rng)
    fut_b, _ = simulate(mu, args.n_sim, L, s, rng)
    _, obs_c = simulate(mu, args.n_obs_avg, L, s, rng)
    query = np.stack([xx.ravel(), yy.ravel()], 1)

    exact_avg = np.zeros((G * G, T))
    for o in obs_c:
        exact_avg += gt.condition(o).grid_density(query).numpy()
    exact_avg /= len(obs_c)
    q4 = torch.tensor(query, dtype=torch.float32)[:, None, None].expand(-1, T, 1, 2)
    old = torch.exp(stored.log_prob(q4[:, :, 0])).numpy()        # (G*G, T)

    pop = {f"{m}_{ref}": [] for m in ("sinkhorn", "sinkhorn_div", "emd") for ref in ("floor", "exact", "old")}
    for n in range(T):
        ha = hist(fut_a[:, n], ex, ey)
        cand = {"floor": hist(fut_b[:, n], ex, ey),
                "exact": exact_avg[:, n].reshape(G, G),
                "old": old[:, n].reshape(G, G)}
        for ref, q in cand.items():
            pop[f"sinkhorn_{ref}"].append(sinkhorn(ha, q, M, dev))
            pop[f"sinkhorn_div_{ref}"].append(sinkhorn_div(ha, q, M, dev))
            pop[f"emd_{ref}"].append(emd_exact(coarse(ha), coarse(q), Mc))
    pop_mean = {k: float(np.mean(v)) for k, v in pop.items()}
    print("population distances to the generator histogram (mean over 12 steps):")
    for k, v in pop_mean.items():
        print(f"   {k:20s} {v:.4f}")

    # ---------------- 2. conditioning check ----------------
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)
    metrics = build_metrics(cfg)
    us = {k: {"step": [], "joint": []} for k in (0, 1, 3, 5)}
    for i in range(len(loader.dataset)):
        dd = metrics.denormalize(dict_to_device(loader.collate_fn([loader.dataset[i]]), dev))
        obs = dd["obs"][0, :, :2].double().cpu()
        g = dd["gt"][0].double().cpu()
        for k in us:
            gt.condition(obs, g[:k] if k else None)
            smp = gt.sample(1000, rng)
            seq = torch.cat([smp, g[k:][None]], 0)
            ls, lj = gt.step_logp(seq), gt.joint_logp(seq)
            us[k]["step"] += ((ls[:-1] > ls[-1]).double().mean(0)).tolist()
            us[k]["joint"].append(float((lj[:-1] > lj[-1]).double().mean()))
    cond = {}
    for k, d in us.items():
        for kind, u in d.items():
            u = np.asarray(u)
            cov = (u[:, None] <= ALPHAS[None]).mean(0)
            cond[f"k={k} {kind}"] = {"ECE": float(np.abs(cov - ALPHAS).mean()),
                                     "signed": float((cov - ALPHAS).mean()), "n": int(len(u))}
            print(f"conditioning k={k} {kind:5s}: ECE {cond[f'k={k} {kind}']['ECE']:.4f} "
                  f"signed {cond[f'k={k} {kind}']['signed']:+.4f} (n={len(u)})")

    out = args.out or f"results/exact_gt/validate_{args.dataset}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"population_sinkhorn_per_step": pop, "population_sinkhorn_mean": pop_mean,
               "conditioning": cond}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
