"""HPD coverage of the fast update (SNIS reweighting) on the 100x100 grid.

The updated density lives on the cached particles only, so it cannot be
evaluated at an arbitrary point. Instead every density is rasterized onto one
shared 100x100 grid per snapshot, exactly as `scripts/eval_update_error.py`
does for its KL/Sinkhorn columns (same routine, same grid extent taken from
the initial forecast), and

    u = sum of the mass in cells denser than the GT cell
        + 0.5 * mass of cells as dense as the GT cell,

with the GT cell the grid node nearest to the ground truth (u = 1 if the
ground truth falls outside the grid). Three densities are scored at the same
absolute steps t+k+1 .. t+T_f:

    pre     the forecast made at t, before any update
    post    the same particles reweighted by update_from_new_obs(., k)
    oracle  a full recomputation from the observation window ending at t+k

Snapshots: the 150 evenly spaced ETH snapshots behind the paper's stratum and
ESS tables (crf_update/gateoff_records_shard*.json), k = 1..6, gate off.
Each row also carries ESS/N of the update and the motion stratum.

    pyenv exec python scripts/calib_update_hpd.py --device cuda:0 --num_shards 4 --shard 0
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "scripts"))

from keepflow.utils import load_config
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics
from keepflow.visualization import build_visualizer
from keepflow.visualization.traj.traj_visualizer import get_grid
from train import dict_to_device
from eval_update_error import (build_shift_lookup, indices_consistent, rotation_for_item,
                               align_to_world, compute_motion_stats, rasterize,
                               effective_sample_size)

RECORDS = REPO_ROOT / "202604_Neurocomputing_FlowChain-CNF" / "crf_update" / "gateoff_records_shard*.json"
FIELDS = ["case_id", "agent", "t", "k", "n", "source", "u", "ess_frac", "stratum",
          "tie_mass", "gt_outside"]


def stratum(ms):
    """The five strata of the paper's Table (assigned in this order)."""
    ta = ms["turn_angle_deg"]
    if ta == ta and ta >= 60.0:
        return "abrupt_turn"
    if ms["displacement"] < 0.10:
        return "stopping"
    r = ms["speed_change_ratio"]
    if r == r and r >= 1.0:
        return "sharp_acceleration"
    if (ms["n_neighbors_within_2m"] or 0) >= 3:
        return "dense_interaction"
    return "steady"


def grid_hpd(zz, xx, yy, gt_xy):
    """zz: (T, G, G) rasterized density (NaN = no support); gt_xy: (T, 2)."""
    zz = np.nan_to_num(np.clip(zz, 0.0, None), nan=0.0)
    flat = zz.reshape(zz.shape[0], -1)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    dx = (xx.max() - xx.min()) / (xx.shape[1] - 1)
    dy = (yy.max() - yy.min()) / (yy.shape[0] - 1)
    out = []
    for t in range(flat.shape[0]):
        g = gt_xy[t]
        outside = (g[0] < xx.min() - dx / 2 or g[0] > xx.max() + dx / 2 or
                   g[1] < yy.min() - dy / 2 or g[1] > yy.max() + dy / 2)
        total = flat[t].sum()
        if outside or total <= 0:
            out.append((1.0, 0.0, outside))
            continue
        p = flat[t] / total
        c = np.argmin(((pts - g) ** 2).sum(1))
        tie = p[p == p[c]].sum()
        out.append((float(p[p > p[c]].sum() + 0.5 * tie), float(tie), False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_file",
                    default="configs/traj/FlowMatchingChain_onestep/CIF_spline/eth.yml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--update_timesteps", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--grid_size", type=int, default=100)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default="results/calib/u_cnf-update_eth.csv")
    args = ap.parse_args()

    cfg = load_config(args)
    cfg.defrost(); cfg.MODEL.FLOW.UPDATE_THRESHOLD = 1000.0; cfg.MODEL.FLOW.ESS_THRESHOLD = 0.0; cfg.freeze()

    recs = []
    for p in sorted(glob.glob(str(RECORDS))):
        recs += json.load(open(p))
    snapshots = sorted({r["scene_idx"] for r in recs})
    assert len(snapshots) == 150, len(snapshots)
    snapshots = snapshots[args.shard::args.num_shards]

    model = build_model(cfg); model.load(); model.eval()
    metrics = build_metrics(cfg)
    visualizer = build_visualizer(cfg)
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)
    dataset, collate_fn = loader.dataset, loader.collate_fn
    shift_item = build_shift_lookup(cfg, dataset)
    pred_len = cfg.DATA.PREDICT_LENGTH
    ks = args.update_timesteps

    out = args.out if args.num_shards == 1 else \
        args.out.replace(".csv", f".shard{args.shard}of{args.num_shards}.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fp, torch.no_grad():
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        for count, idx in enumerate(snapshots, 1):
            torch.manual_seed(cfg.SEED + idx)
            np.random.seed(cfg.SEED + idx)
            item = dataset[idx]
            dd = dict_to_device(collate_fn([item]), cfg.DEVICE)
            index0 = dd["index"][0]
            st = stratum(compute_motion_stats(dd))

            result0 = model.predict(dd, return_prob=True)
            ess = {}
            for k in ks:
                ess[k] = effective_sample_size(model, result0, k)["ess_frac"]
                result0 = model.update_from_new_obs(result0, k)
            result0 = metrics.denormalize(result0)
            result0 = align_to_world(result0, rotation_for_item(cfg, item, cfg.DEVICE))
            min_pos, max_pos = visualizer.get_minmax(result0, visualize=False)
            xx, yy = get_grid(args.grid_size, min_pos, max_pos)

            for k in ks:
                oi = shift_item(idx, k)
                if oi is None:
                    continue
                od = dict_to_device(collate_fn([oi]), cfg.DEVICE)
                if not indices_consistent(index0, od["index"][0], k):
                    continue
                orc = model.predict(od, return_prob=True)
                orc = metrics.denormalize(orc)
                orc = align_to_world(orc, rotation_for_item(cfg, oi, cfg.DEVICE))

                n_ov = pred_len - k
                gt = result0["gt"][0, k:k + n_ov].cpu().numpy()
                dens = {"pre": result0[("prob", 0)][:, :, k:],
                        "post": result0[("prob", k)][:, :, :n_ov],
                        "oracle": orc[("prob", 0)][:, :, :n_ov]}
                for src, prob in dens.items():
                    zz = rasterize(visualizer, prob, xx, yy, min_pos, max_pos)
                    for j, (u, tie, outside) in enumerate(grid_hpd(zz, xx, yy, gt)):
                        w.writerow({"case_id": idx, "agent": index0[2], "t": int(index0[1]),
                                    "k": k, "n": k + j + 1, "source": src,
                                    "u": f"{u:.6f}", "ess_frac": f"{ess[k]:.6f}",
                                    "stratum": st, "tie_mass": f"{tie:.6f}",
                                    "gt_outside": int(outside)})
            fp.flush()
            print(f"snapshot {count}/{len(snapshots)} (idx {idx}, {st})", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
