"""Adaptive-solver NFE: does the spline (CRF) target make the field cheaper to integrate?

The fixed-step sweep (`run_nfe_sweep.py`) cannot answer this. There the solver's
steps are built per segment (`get_indices_t_span` concatenates
`linspace(i, i+1, num_step)`), so a step never straddles an integer knot -- which
is exactly the configuration in which the piecewise-linear target's velocity
discontinuities at the knots cost nothing. That sweep therefore measures path
curvature only, and a straight path wins by construction.

The default solver is `dopri5` (adaptive, tol 1e-2) integrating across the whole
[0, T_f] span, where internal steps *do* cross the knots. There a kinked field
forces step rejections and step-size reduction, while a C2 field lets the solver
keep its order. The number of function evaluations is both the direct inference
cost and the sharpest available probe of field regularity.

Reports NFE per prediction (mean +/- std over the test set) for each arm.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keepflow.utils import load_config
from keepflow.data import build_dataloader
from keepflow.models import build_model
from train import dict_to_device


class NFECounter:
    """Count vector-field evaluations by hooking the field itself.

    torchdyn exposes no evaluation counter (the `self.nde[1].nfe = 0` line in
    `CNF_step.forward` writes a stray attribute that nothing reads), so the
    count is taken where it is unambiguous: one forward of the vector field is
    one function evaluation by the solver.
    """

    def __init__(self, model):
        self.n = 0
        self.handles = []
        fields = [m for name, m in model.named_modules()
                  if name.endswith("vector_field")]
        if not fields:
            raise SystemExit("no `vector_field` submodule found -- is this a CNF model?")
        for f in fields:
            self.handles.append(f.register_forward_hook(self._hook))
        self.n_fields = len(fields)

    def _hook(self, *_):
        self.n += 1

    def reset(self):
        self.n = 0

    def close(self):
        for h in self.handles:
            h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_scenes", type=int, default=200)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args)
    assert cfg.MODEL.FLOW.CNF_SOLVER in ("dopri5", "tsit5"), \
        f"NFE is only meaningful for an adaptive solver, got {cfg.MODEL.FLOW.CNF_SOLVER}"

    model = build_model(cfg)
    model.load()
    model.eval()

    counter = NFECounter(model)

    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)
    n = min(args.n_scenes, len(loader.dataset))

    per_pred = []
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= n:
                break
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            counter.reset()
            model.predict(data_dict, return_prob=True)
            per_pred.append(counter.n)
    counter.close()

    per_pred = np.array(per_pred, dtype=float)
    res = {"config": args.config_file,
           "solver": cfg.MODEL.FLOW.CNF_SOLVER,
           "n_predictions": int(per_pred.size),
           "nfe_mean": float(per_pred.mean()),
           "nfe_std": float(per_pred.std(ddof=1)) if per_pred.size > 1 else 0.0,
           "nfe_median": float(np.median(per_pred)),
           "nfe_min": float(per_pred.min()), "nfe_max": float(per_pred.max())}
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fp:
            json.dump(res, fp, indent=2)


if __name__ == "__main__":
    main()
