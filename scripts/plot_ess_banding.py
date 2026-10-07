"""Where the fast density update stops being a sound approximation .

The update is self-normalized importance reweighting of a cached particle set. When the
new observation is far from what the cached particles predicted, almost all the weight
lands on a handful of particles and the effective sample size collapses; the density is
then supported by a few particles that happen to lie near the observation, which is why it
can be both very confident and very wrong.

This plots the error against a fully recomputed oracle as a function of ESS/N, over the
900 gate-free records (150 ETH snapshots x k=1..6). It shows the one thing the k-based
view could not: the damage is confined to ESS/N < 0.05, and k only makes degeneracy more
likely rather than determining it.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "202604_Neurocomputing_FlowChain-CNF")
RECORDS = os.path.join(PAPER, "crf_update", "gateoff_records_shard*.json")
OUT = os.path.join(PAPER, "maeda12.pdf")

BANDS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20),
         (0.20, 0.30), (0.30, 0.50), (0.50, 1.01)]
THRESHOLD = 0.05


def load():
    recs = []
    for p in sorted(glob.glob(RECORDS)):
        recs += json.load(open(p))
    return [r for r in recs
            if r.get("status") == "ok" and r.get("ess_frac") is not None]


def main():
    ok = load()
    ess = np.array([r["ess_frac"] for r in ok])
    diff = np.array([r["logprob_gt_diff_mean"] for r in ok])
    k = np.array([r["k"] for r in ok])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.55, 1]})

    # ---- left: per-record error against ESS, with the band summary on top ----
    ax.axhline(0, color="0.35", lw=.8, zorder=1)
    ax.axvspan(0, THRESHOLD, color="tab:red", alpha=.08, zorder=0)
    sc = ax.scatter(ess, diff, c=k, cmap="viridis", s=11, alpha=.75,
                    linewidths=0, zorder=3)
    for lo, hi in BANDS:
        m = (ess >= lo) & (ess < hi)
        if not m.any():
            continue
        ax.plot([lo, hi], [diff[m].mean()] * 2, color="black", lw=2.0, zorder=4)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"effective sample size $\mathrm{ESS}/N$")
    ax.set_ylabel("GT log-probability,\nupdate minus full recomputation")
    ax.annotate("degenerate\nweights", xy=(0.025, 0.90),
                xycoords=("data", "axes fraction"),
                ha="left", va="top", fontsize=8, color="tab:red")
    cb = fig.colorbar(sc, ax=ax, pad=.02, aspect=28)
    cb.set_label("update horizon $k$", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.tick_params(labelsize=9)

    # ---- right: tail risk per band ----
    # The mean hides what matters. A fraction-worse-than-oracle bar would read 40% in the
    # top band too, because near ESS/N = 1 the update and the oracle agree to within a
    # fraction of a nat and the sign of a negligible difference is arbitrary. The 10th
    # percentile is the quantity a practitioner needs: how bad it gets when it goes bad.
    centres = [(lo + min(hi, 1.0)) / 2 for lo, hi in BANDS]
    widths = [min(hi, 1.0) - lo for lo, hi in BANDS]
    p10, ns = [], []
    for lo, hi in BANDS:
        m = (ess >= lo) & (ess < hi)
        p10.append(np.percentile(diff[m], 10) if m.any() else np.nan)
        ns.append(int(m.sum()))
    colours = ["tab:red" if lo < THRESHOLD else "0.6" for lo, _ in BANDS]
    ax2.bar(centres, p10, width=[w * .9 for w in widths],
            color=colours, edgecolor="white", linewidth=.6)
    # Per-band record counts are not printed here; they only crowd the three
    # narrow bands.
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-200, 0)
    ax2.set_yticks([0, -1, -10, -100])
    ax2.set_xlabel(r"$\mathrm{ESS}/N$")
    ax2.set_ylabel("worst decile of the same\ndifference ($p_{10}$, nats)")
    ax2.tick_params(labelsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=300)
    print(f"wrote {OUT} (+ .png)")

    keep = ess >= THRESHOLD
    print(f"  ESS/N >= {THRESHOLD}: kept {keep.mean():.1%} of {len(ok)} records, "
          f"mean {diff[keep].mean():+.2f}; dropped mean {diff[~keep].mean():+.2f}")


if __name__ == "__main__":
    main()
