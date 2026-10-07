"""Accuracy-vs-runtime Pareto plot for the ETH density estimation comparison.

The KDE baselines expose a sample budget N that trades runtime against density
accuracy; FlowChain has no such knob because its density is analytic. Plotting
every measured (time, log-probability) point makes two things visible at once:
FlowChain-CNF sits on the Pareto front, and every baseline's accuracy *stops
improving* -- and then degrades -- as N grows, because Scott's-rule bandwidth
shrinks with N and the ground truth starts falling between the samples.

Numbers come from the corrected protocol: all 44 ETH agents (not the first 9,
which are only 5 distinct pedestrians), log-sum-exp KDE (no underflow floor),
and the analytic density for FlowChain (no grid interpolation).
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "experiments/configs/traj")
BUDGETS = [20, 100, 1000, 10000]
BASELINES = [
    ("socialGAN", "Social-GAN", "tab:blue", "o"),
    ("SingularTrajectory", "SingularTrajectory", "tab:orange", "s"),
    ("MID", "MID", "tab:green", "^"),
    ("MoFlow/fm", "MoFlow", "tab:red", "D"),
]


def read(path, key):
    if not os.path.exists(path):
        return None
    v = json.load(open(path)).get(key)
    if v is None:
        return None
    m = re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(v))
    return float(m[0]) if m else None


def main():
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    pts = []

    for key, label, colour, marker in BASELINES:
        xs, ys = [], []
        for n in BUDGETS:
            f = f"{BASE}/{key}/eth/metrics_sweep2_n{n}.json"
            t, lp = read(f, "execution time"), read(f, "log_prob_mean")
            if t is None or lp is None:
                continue
            xs.append(t)
            ys.append(lp)
            pts.append((t, lp, label))
        if not xs:
            continue
        ax.plot(xs, ys, "-", color=colour, alpha=.55, lw=1.3, zorder=2)
        ax.scatter(xs, ys, s=46, color=colour, marker=marker, label=label,
                   zorder=3, edgecolors="white", linewidths=.7)
        # annotate the budget at the two ends only, to keep the plot readable
        for x, y, n in [(xs[0], ys[0], BUDGETS[0]), (xs[-1], ys[-1], BUDGETS[len(xs) - 1])]:
            ax.annotate(f"N={n:,}", (x, y), textcoords="offset points",
                        xytext=(4, -11), fontsize=7.5, color=colour)

    # FlowChain-CNF only. Its log-probability comes from the analytic density,
    # evaluated exactly at the ground truth, which is the footing the KDE
    # baselines and Trajectron++ are scored on too. FlowChain-NF is
    # autoregressive: each step is trained against a base distribution centred on
    # the previous position, so the density of a step given only the observation
    # window has no closed form and the model cannot be placed on this axis.
    CNF_TIME_MS, CNF_LOGPROB = 195.0, -0.741   # 44 ETH agents, three seeds
    print(f"  FlowChain-CNF: {CNF_LOGPROB:+.3f} @ {CNF_TIME_MS:.0f} ms")
    ax.scatter([CNF_TIME_MS], [CNF_LOGPROB], s=210, color="black", marker="*",
               label="FlowChain-CNF (ours)", zorder=5,
               edgecolors="white", linewidths=.8)
    ax.annotate("analytic density\n(no sample budget)", (CNF_TIME_MS, CNF_LOGPROB),
                textcoords="offset points", xytext=(9, 4), fontsize=8, color="black")
    pts.append((CNF_TIME_MS, CNF_LOGPROB, "FlowChain-CNF (ours)"))

    # Pareto front: a point is dominated if another is both faster and better.
    front = sorted((p for p in pts
                    if not any(o[0] < p[0] and o[1] > p[1] for o in pts)),
                   key=lambda p: p[0])
    ax.plot([p[0] for p in front], [p[1] for p in front], "--",
            color="black", lw=1.1, alpha=.5, zorder=1, label="Pareto front")

    ax.set_xscale("log")
    ax.set_xlabel("inference time per scene [ms, log scale]")
    ax.set_ylabel("ETH ground-truth log-probability  (higher is better)")
    ax.grid(alpha=.25, which="both", ls=":")
    # outside the axes: the baselines' cheap, low-accuracy points live in the
    # lower-left corner and an inset legend hides them
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5,
              framealpha=.95, borderaxespad=0)
    fig.tight_layout()

    out = os.path.join(REPO, "202604_Neurocomputing_FlowChain-CNF",
                       "maeda5.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=170)
    print(f"wrote {out} (+ .png)")
    print("Pareto front:")
    for t, lp, label in front:
        print(f"  {label:28s} {t:9.0f} ms   {lp:+.3f}")


if __name__ == "__main__":
    main()
