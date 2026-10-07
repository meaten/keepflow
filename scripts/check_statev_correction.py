"""Is the velocity-to-position correction applied once, or once per step?

FlowChain-NF predicts velocities, so its density is over velocity and has to be
converted before it can be compared with position-space densities. For a single
step the map is x_n = x_{n-1} + dt * v_n, whose Jacobian in 2-D is dt^2, giving
-2 log dt = 1.833 nats at dt = 0.4. `analytic_log_prob` applies that once.

The chained density runs n transformations before the conversion, so the question
is whether the correction should scale with n. This compares the chained value
against an independent position-space density of the same model: a kernel density
estimate over the model's own particles, which are integrated into positions
before the estimate is fitted. The KDE is used only as a yardstick here, not as a
proposal for how the paper should report anything.

If the correction should be n-fold, the chained value will drift upward against
the KDE by about 1.833 nats per step.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keepflow.data import build_dataloader
from keepflow.metrics import build_metrics
from keepflow.models import build_model
from keepflow.utils import load_config, kde
from train import dict_to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_scenes", type=int, default=9)
    args = p.parse_args()

    cfg = load_config(args)
    model = build_model(cfg)
    model.load()
    model.eval()
    metrics = build_metrics(cfg)
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)

    C, K = [], []
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            if i >= args.n_scenes:
                break
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            chained = model.analytic_log_prob_chain(data_dict, [0])[0]

            res = model.predict(data_dict, return_prob=True)
            res = metrics.denormalize(res)          # particles -> positions
            res = kde(res)                          # position-space density
            k = res[("gt_traj_log_prob", 0)].reshape(-1).cpu().numpy()

            n = min(len(k), chained.numel())
            C.append(chained.reshape(-1).cpu().numpy()[:n])
            K.append(k[:n])

    T = len(C[0])
    Cm = np.stack(C).reshape(-1, T)
    Km = np.stack(K).reshape(-1, T)
    ok = Km > -999
    print(f"scenes {Cm.shape[0]}, steps {T}")
    print("step   chained      KDE     diff   diff/step")
    for t in range(T):
        m = ok[:, t]
        if m.sum() == 0:
            print(f"{t+1:4d}        --       --       --")
            continue
        d = Cm[m, t].mean() - Km[m, t].mean()
        print(f"{t+1:4d} {Cm[m, t].mean():9.2f} {Km[m, t].mean():8.2f} {d:8.2f} "
              f"{d / (t + 1):10.2f}")
    d_all = Cm[ok] - Km[ok]
    slope = np.polyfit(np.tile(np.arange(1, T + 1), Cm.shape[0])[ok.reshape(-1)],
                       d_all, 1)[0]
    print(f"\nmean diff {d_all.mean():.2f}; slope of diff against step {slope:.3f} "
          f"(1.833 would mean the correction should be n-fold, 0 that once is right)")


if __name__ == "__main__":
    main()
