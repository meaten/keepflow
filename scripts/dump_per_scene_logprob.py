"""Per-scene, per-step ground-truth log-probability.

The reported `log_prob_*` metrics are means over the evaluated scenes, and a
log-probability mean is dominated by its tail: a single scene where the model
puts almost no mass on the ground truth can move the mean by many nats. This
dumps the underlying per-scene values so that the mean can be checked against
its distribution (and against the two flooring mechanisms: the explicit
`nan_to_num(neginf=-10000)` in `keepflow/utils/kde.py`, and the implicit float32
underflow at log(1.4e-45) = -103.28).
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
from train import dict_to_device

FLOAT32_UNDERFLOW_LOG = -103.2789306640625


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_scenes", type=int, default=9)
    p.add_argument("--fail_threshold", type=float, default=-10.0,
                   help="a cell counts as a density failure when the ground-truth "
                        "log-probability falls below this (default -10, i.e. the "
                        "predicted density at the ground truth is below 4.5e-5)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    metrics = build_metrics(cfg)
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    rows = []
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= args.n_scenes:
                break
            idx = data_dict["index"][0]
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            res = model.predict(data_dict, return_prob=True)
            res = metrics.denormalize(res)
            if cfg.TEST.KDE:
                res = kde(res)
            lp = res.get(("gt_traj_log_prob", 0))
            if lp is None:
                print("model did not produce gt_traj_log_prob")
                return
            v = lp.detach().cpu().numpy().reshape(-1)
            rows.append({"i": i, "scene": str(idx[0]), "t": int(idx[1]),
                         "agent": str(idx[2]), "logp": v.tolist()})
            print(f"[{i}] {idx[0]}/{idx[2]} t={int(idx[1]):4d}  "
                  f"per-step logp = {np.array2string(v, precision=2, max_line_width=250)}",
                  flush=True)

    if not rows:
        return
    A = np.array([r["logp"] for r in rows])          # (scenes, steps)
    stats = tail_statistics(A, args.fail_threshold)

    print(f"\nshape = {A.shape} (scenes x steps)")
    print(f"per-step mean = {np.array2string(A.mean(axis=0), precision=2, max_line_width=250)}")
    print(f"per-step p10  = {np.array2string(np.percentile(A, 10, axis=0), precision=2, max_line_width=250)}")
    print("\n--- headline (mean) and the tail, which is where an analytic density "
          "differs from a post-hoc KDE ---")
    print(f"  mean log-prob                      : {stats['mean']:12.3f}")
    print(f"  10th percentile (worst decile)     : {stats['p10']:12.3f}")
    print(f"  25th percentile                    : {stats['p25']:12.3f}")
    print(f"  failure rate P(log p < {args.fail_threshold:g})        : "
          f"{stats['failure_rate']:12.4f}  ({stats['n_fail']}/{A.size} cells)")
    print(f"  minimum                            : {stats['min']:12.3f}")
    print("\n--- numerical-floor audit (should now be zero everywhere) ---")
    print(f"  cells at/below the float32 subnormal floor (-103.28): {stats['n_underflow']}/{A.size}")
    print(f"  cells at the explicit -10000 clamp                  : {stats['n_hard_floor']}/{A.size}")

    if args.out:
        with open(args.out, "w") as fp:
            json.dump({"config": args.config_file, "records": rows, "stats": stats}, fp, indent=2)


def tail_statistics(A: np.ndarray, fail_threshold: float) -> dict:
    """Summaries that keep the tail visible.

    A robust statistic (median, trimmed mean) would defeat the purpose here: the
    whole difference between an analytic density and a post-hoc KDE shows up in
    the tail, where the KDE has no samples and assigns essentially zero density.
    So the mean is kept, and it is accompanied by statistics that probe the tail
    *without* being set by the magnitude of a single catastrophic cell: a low
    percentile, and the rate at which the density loses the ground truth
    altogether.
    """
    return {
        "mean": float(A.mean()),
        "p10": float(np.percentile(A, 10)),
        "p25": float(np.percentile(A, 25)),
        "min": float(A.min()),
        "failure_rate": float((A < fail_threshold).mean()),
        "n_fail": int((A < fail_threshold).sum()),
        "n_cells": int(A.size),
        "n_underflow": int((A <= FLOAT32_UNDERFLOW_LOG + 1.0).sum()),
        "n_hard_floor": int((A <= -9999).sum()),
    }


if __name__ == "__main__":
    main()
