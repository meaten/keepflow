"""Reliability diagrams, ECE and bootstrap CIs from the u values of calib_hpd.py.

    coverage(a) = mean 1[u <= a],  a in {0.05, 0.10, ..., 0.95}
    ECE         = mean_a |coverage(a) - a|
    signed      = mean_a (coverage(a) - a)    (< 0: overconfident)

CIs are 95% percentile intervals over a bootstrap of test cases (B = 2000).
For the step-pooled row (n = "all") a case is resampled together with all of
its steps, since the 12 steps of one case are not independent.

Merges shard files (u_<m>_<d>.shardKofN.csv) into u_<m>_<d>.csv first.

    pyenv exec python scripts/calib_summary.py
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHAS = np.round(np.arange(0.05, 0.951, 0.05), 2)
B = 2000

DATASETS = ["simunicycle", "simfork", "simbranch", "eth", "hotel", "univ",
            "zara1", "zara2", "sdd"]
LABEL = {"simunicycle": "Unicycle", "simfork": "Fork", "simbranch": "Branch",
         "eth": "ETH", "hotel": "Hotel", "univ": "Univ", "zara1": "Zara1",
         "zara2": "Zara2", "sdd": "SDD"}
# reference categorical order (dataviz palette, light mode); GT is a neutral reference
STYLE = {
    "cnf":        dict(color="#2a78d6", label="FlowChain-CNF (per-step marginal, pooled)"),
    "cnf-last":   dict(color="#2a78d6", ls="--", label="FlowChain-CNF ($n=T_f$)"),
    "nf":         dict(color="#eb6834", label="FlowChain-NF (joint, ancestral)"),
    "nf-native":  dict(color="#1baf7a", label="FlowChain-NF (joint, native sampler)"),
    "trajectron": dict(color="#eda100", label="Trajectron++ (per-step, pooled)"),
    "gt":         dict(color="#7a7974", ls=":", label="GT posterior (validation)"),
}


def merge_shards(d):
    groups = {}
    for p in glob.glob(os.path.join(d, "u_*.shard*of*.csv")):
        stem = re.sub(r"\.shard\d+of\d+\.csv$", "", os.path.basename(p))
        groups.setdefault(stem, []).append(p)
    for stem, paths in groups.items():
        m = int(re.search(r"of(\d+)\.csv$", paths[0]).group(1))
        if len(paths) != m:
            print(f"  {stem}: {len(paths)}/{m} shards present, not merged yet")
            continue
        df = pd.concat([pd.read_csv(p) for p in sorted(paths)])
        df = df.sort_values(["case_id", "n"])
        df.to_csv(os.path.join(d, f"{stem}.csv"), index=False)


def coverage(u):
    return (u[:, None] <= ALPHAS[None]).mean(0)


def stats(cov):
    return np.abs(cov - ALPHAS).mean(), (cov - ALPHAS).mean()


def bootstrap(df, rng):
    """df: rows of one (method, dataset, n-group). Returns point and CI dicts."""
    cases = df["case_id"].unique()
    by_case = {c: g["u"].to_numpy() for c, g in df.groupby("case_id")}
    cov = coverage(df["u"].to_numpy())
    ece, sgn = stats(cov)
    eces, sgns, covs = [], [], []
    for _ in range(B):
        pick = rng.choice(cases, size=len(cases), replace=True)
        u = np.concatenate([by_case[c] for c in pick])
        c = coverage(u)
        e, s = stats(c)
        eces.append(e); sgns.append(s); covs.append(c)
    covs = np.array(covs)
    return dict(cov=cov, cov_lo=np.percentile(covs, 2.5, 0), cov_hi=np.percentile(covs, 97.5, 0),
                ece=ece, ece_lo=np.percentile(eces, 2.5), ece_hi=np.percentile(eces, 97.5),
                signed=sgn, signed_lo=np.percentile(sgns, 2.5), signed_hi=np.percentile(sgns, 97.5),
                n_cases=len(cases))


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "u_*_*.csv")):
        name = os.path.basename(p)
        if ".shard" in name:
            continue
        m = re.match(r"u_(.+)_([a-z0-9]+)\.csv$", name)
        if m is None or m.group(2) not in DATASETS or m.group(1).endswith("-update"):
            continue
        out[(m.group(1), m.group(2))] = pd.read_csv(p)
    return out


def groups_for(method, df):
    """Yield (n_label, rows) for every summary row of one file."""
    ns = sorted(df["n"].unique())
    for n in ns:
        yield str(n) if n != -1 else "joint", df[df["n"] == n]
    if len([n for n in ns if n > 0]) > 1:
        yield "all", df[df["n"] > 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/calib")
    args = p.parse_args()
    merge_shards(args.dir)
    data = load(args.dir)
    rng = np.random.default_rng(0)

    rows, curves = [], {}
    for (method, ds), df in sorted(data.items()):
        for n_label, g in groups_for(method, df):
            r = bootstrap(g, rng)
            curves[(method, ds, n_label)] = r
            rows.append({"method": method, "dataset": ds, "n": n_label,
                         "n_cases": r["n_cases"],
                         "ECE": r["ece"], "ECE_lo": r["ece_lo"], "ECE_hi": r["ece_hi"],
                         "signed": r["signed"], "signed_lo": r["signed_lo"],
                         "signed_hi": r["signed_hi"]})
            print(f"{method:10s} {ds:12s} n={n_label:5s} cases={r['n_cases']:5d} "
                  f"ECE={r['ece']:.3f} [{r['ece_lo']:.3f},{r['ece_hi']:.3f}] "
                  f"signed={r['signed']:+.3f} [{r['signed_lo']:+.3f},{r['signed_hi']:+.3f}]")
    upd_path = os.path.join(args.dir, "u_cnf-update_eth.csv")
    if os.path.exists(upd_path):
        rows += summarize_update(pd.read_csv(upd_path), args.dir, rng)
    pd.DataFrame(rows).to_csv(os.path.join(args.dir, "summary.csv"), index=False,
                              float_format="%.4f")

    plot_reliability(curves, data, args.dir)
    plot_step_ece(curves, args.dir)
    plot_pit(data, args.dir)


UPD_STYLE = {"oracle": dict(color="#7a7974", ls=":", label="oracle (full recomputation)"),
             "pre": dict(color="#eb6834", label="before update"),
             "post": dict(color="#2a78d6", label="after update (SNIS)")}
STRATA = [("steady", "Steady motion"), ("dense_interaction", "Dense interaction"),
          ("stopping", "Stopping"), ("abrupt_turn", "Abrupt turn"),
          ("sharp_acceleration", "Sharp acceleration")]


def summarize_update(df, d, rng):
    """Grid-HPD reliability of the update path; bootstrap over snapshots."""
    groups = [("all", df), ("ESS<0.05", df[df.ess_frac < 0.05]),
              ("ESS>=0.05", df[df.ess_frac >= 0.05])]
    groups += [(f"stratum:{k}", df[df.stratum == k]) for k, _ in STRATA]
    rows, curves = [], {}
    for gname, g in groups:
        for src in ("oracle", "pre", "post"):
            gs = g[g.source == src]
            if len(gs) == 0:
                continue
            r = bootstrap(gs, rng)
            curves[(gname, src)] = (r, len(gs))
            rows.append({"method": f"cnf-update:{src}", "dataset": "eth", "n": "all",
                         "group": gname, "n_cases": r["n_cases"], "n_rows": len(gs),
                         "ECE": r["ece"], "ECE_lo": r["ece_lo"], "ECE_hi": r["ece_hi"],
                         "signed": r["signed"], "signed_lo": r["signed_lo"],
                         "signed_hi": r["signed_hi"],
                         "tie_mass_mean": gs.tie_mass.mean(),
                         "gt_outside_frac": gs.gt_outside.mean()})
            print(f"update {src:6s} {gname:28s} snaps={r['n_cases']:3d} rows={len(gs):5d} "
                  f"ECE={r['ece']:.3f} [{r['ece_lo']:.3f},{r['ece_hi']:.3f}] "
                  f"signed={r['signed']:+.3f} [{r['signed_lo']:+.3f},{r['signed_hi']:+.3f}]")
    for fname, panels in [("update_reliability_ess", [("ESS<0.05", r"$\mathrm{ESS}/N<0.05$"),
                                                      ("ESS>=0.05", r"$\mathrm{ESS}/N\geq0.05$"),
                                                      ("all", "all records")]),
                          ("update_reliability_strata", [(f"stratum:{k}", lab) for k, lab in STRATA])]:
        fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 3.0), squeeze=False)
        handles = {}
        for ax, (gname, title) in zip(axes[0], panels):
            _diag(ax)
            n_rows = None
            for src in ("oracle", "pre", "post"):
                if (gname, src) not in curves:
                    continue
                r, n_rows = curves[(gname, src)]
                st = dict(UPD_STYLE[src]); lab = st.pop("label")
                ax.fill_between(ALPHAS, r["cov_lo"], r["cov_hi"], color=st["color"], alpha=0.15, lw=0)
                handles[lab], = ax.plot(ALPHAS, r["cov"], lw=1.6, **st)
                n_snap = r["n_cases"]
            if n_rows is not None:
                ax.set_title(f"{title}\n({n_snap} snapshots, {n_rows} rows)", fontsize=8)
            ax.set_xlabel(r"nominal level $\alpha$", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0, 0].set_ylabel(r"coverage $P(u\leq\alpha)$", fontsize=8)
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3,
                   fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(d, f"{fname}.{ext}"), dpi=200)
        plt.close(fig)
    return rows


def _diag(ax):
    ax.plot([0, 1], [0, 1], color="#0b0b0b", lw=0.8, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(color="#e4e3df", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_reliability(curves, data, d):
    present = [ds for ds in DATASETS if any(k[1] == ds for k in data)]
    ncol = min(5, len(present))
    nrow = int(np.ceil(len(present) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow + 0.6),
                             squeeze=False)
    for ax in axes.flat[len(present):]:
        ax.axis("off")
    handles = {}
    for ax, ds in zip(axes.flat, present):
        _diag(ax)
        for key, nl in [("gt", "all"), ("gt", "joint"), ("cnf", "all"), ("cnf", "12"),
                        ("trajectron", "all"), ("nf", "joint"), ("nf-native", "joint")]:
            r = curves.get((key, ds, nl))
            if r is None:
                continue
            st = dict(STYLE["cnf-last" if (key, nl) == ("cnf", "12") else key])
            lab = st.pop("label")
            if key == "gt":
                lab = "GT posterior, " + ("joint" if nl == "joint" else "per-step")
                st["ls"] = ":" if nl == "all" else "-."
            ax.fill_between(ALPHAS, r["cov_lo"], r["cov_hi"], color=st["color"],
                            alpha=0.15, lw=0)
            h, = ax.plot(ALPHAS, r["cov"], lw=1.6, **st)
            handles[lab] = h
        ax.set_title(LABEL[ds], fontsize=10)
        ax.set_xlabel(r"nominal level $\alpha$", fontsize=8)
        ax.tick_params(labelsize=7)
    for r in range(nrow):
        axes[r, 0].set_ylabel(r"coverage $P(u\leq\alpha)$", fontsize=8)
    fig.legend(handles.values(), handles.keys(), loc="lower center",
               ncol=min(4, len(handles)), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.6 / (2.5 * nrow + 0.6) + 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"reliability.{ext}"), dpi=200)
    plt.close(fig)


def plot_step_ece(curves, d):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
              "#4a3aa7", "#e34948", "#52514e"]
    for ax, key, title in [(axes[0], "ece", "ECE"), (axes[1], "signed", "signed error")]:
        for c, ds in zip(colors, DATASETS):
            ns = [n for n in range(1, 13) if ("cnf", ds, str(n)) in curves]
            if not ns:
                continue
            v = [curves[("cnf", ds, str(n))][key] for n in ns]
            ax.plot(ns, v, marker="o", ms=3, lw=1.4, color=c, label=LABEL[ds])
        ax.set_xlabel("prediction step $n$", fontsize=8)
        ax.set_title(f"FlowChain-CNF {title} per step", fontsize=9)
        ax.grid(color="#e4e3df", lw=0.5)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].axhline(0, color="#0b0b0b", lw=0.8)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"cnf_step_ece.{ext}"), dpi=200)
    plt.close(fig)


def plot_pit(data, d):
    methods = [m for m in ("cnf", "nf", "nf-native", "trajectron") if any(k[0] == m for k in data)]
    present = [ds for ds in DATASETS if any(k[1] == ds for k in data)]
    fig, axes = plt.subplots(len(methods), len(present),
                             figsize=(1.8 * len(present), 1.5 * len(methods)), squeeze=False)
    for i, m in enumerate(methods):
        for j, ds in enumerate(present):
            ax = axes[i, j]
            df = data.get((m, ds))
            if df is not None:
                u = df["u"].to_numpy() if m in ("nf", "nf-native") else df[df["n"] > 0]["u"].to_numpy()
                ax.hist(u, bins=20, range=(0, 1), density=True,
                        color=STYLE[m]["color"], edgecolor="white", lw=0.5)
            ax.axhline(1, color="#0b0b0b", lw=0.8)
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(LABEL[ds], fontsize=8)
            if j == 0:
                ax.set_ylabel(m, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"pit_hist.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
