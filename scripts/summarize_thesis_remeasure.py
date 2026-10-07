"""Tables for the thesis re-measurement (scripts/run_thesis_remeasure.py).

Every cell is printed as  paper | old GT now | exact GT now  so the change
attributable to the ground truth can be read apart from the drift of the
evaluation code since February. Ablation: mean +/- sample std (ddof=1) over
three seeds, the paper's convention.

    pyenv exec python scripts/summarize_thesis_remeasure.py > results/thesis_remeasure.md
"""
import json
import os
import re

import numpy as np

E = "experiments"
FMC = "configs/traj/FlowMatchingChain_onestep"


def vals(path, key):
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    if key not in m:
        return None
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:e[-+]?\d+)?|[-+]?\d+", m[key])]


def p(cfgdir, ds, tag):
    return f"{E}/{cfgdir}/{ds}/metrics_{tag}.json"


def fmt(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def ms(v, d=3):
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return "n/a"
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}"


MAIN = [("Social-GAN", "configs/traj/socialGAN", 1.46, 1.28),
        ("Trajectron++", "configs/traj/Trajectron", 2.02, 1.67),
        ("MID", "configs/traj/MID", 1.80, 0.97),
        ("SingularTrajectory", "configs/traj/SingularTrajectory", 5.52, 5.04),
        ("MoFlow", "configs/traj/MoFlow/fm", 3.65, 3.08),
        ("FlowChain-NF", "configs/traj/FlowChain/CIF_separate_cond_v", 1.62, 1.44),
        ("FlowChain-CNF (Pred at t)", f"{FMC}/CIF_spline", 1.40, 1.19)]
CNF_UPD = [("Update from t+1", 1, 1.03, 1.43), ("Update from t+3", 2, 0.62, 1.07),
           ("Update from t+5", 3, 0.61, 1.17)]


def main_table():
    print("## Main table: Sinkhorn (Fork / Branch)\n")
    print("paper | old GT now | exact GT now\n")
    print("| Method | Fork | Branch |\n|---|---|---|")
    ranks = {"paper": {}, "old": {}, "exact": {}}
    for name, d, pf, pb in MAIN:
        cells = []
        for ds, pv in (("simfork", pf), ("simbranch", pb)):
            o = vals(p(d, ds, "thesis_exactgt"), "emd")
            x = vals(p(d, ds, "thesis_exactgt"), "emd_exact")
            cells.append(f"{pv:.2f} \\| {fmt(o and o[0])} \\| {fmt(x and x[0])}")
            ranks["paper"].setdefault(ds, {})[name] = pv
            if o:
                ranks["old"].setdefault(ds, {})[name] = o[0]
            if x:
                ranks["exact"].setdefault(ds, {})[name] = x[0]
        print(f"| {name} | {cells[0]} | {cells[1]} |")
    d = f"{FMC}/CIF_spline"
    for name, i, pf, pb in CNF_UPD:
        cells = []
        for ds, pv in (("simfork", pf), ("simbranch", pb)):
            o = vals(p(d, ds, "thesis_exactgt"), "emd")
            x = vals(p(d, ds, "thesis_exactgt"), "emd_exact")
            cells.append(f"{pv:.2f} \\| {fmt(o and o[i])} \\| {fmt(x and x[i])}")
        print(f"| CNF {name} | {cells[0]} | {cells[1]} |")
    print("\n### Ranking of the seven Pred-at-t rows (1 = best)\n")
    for ds in ("simfork", "simbranch"):
        for src in ("paper", "old", "exact"):
            r = ranks[src].get(ds, {})
            order = sorted(r, key=r.get)
            print(f"- {ds} {src:5s}: " + " < ".join(f"{n} ({r[n]:.2f})" for n in order))
    print()


def eth_logprob():
    print("## ETH log-probability, dopri5 (paper column was RK4)\n")
    seeds = [f"{FMC}/CIF_spline", f"configs_seed/s2/{FMC}/CIF_spline", f"configs_seed/s3/{FMC}/CIF_spline"]
    rk4 = [f"{E}/{FMC}/CIF_spline/eth/metrics_chain_s1.json",
           f"{E}/configs_seed/s2/{FMC}/CIF_spline/eth/metrics_chain_s2.json",
           f"{E}/configs_seed/s3/{FMC}/CIF_spline/eth/metrics_chain_s3.json"]
    dop = [vals(p(d, "eth", "thesis_dopri5_chain"), "log_prob_mean") for d in seeds]
    old = [vals(f, "log_prob_mean") for f in rk4]
    print("| Row | paper (RK4) | dopri5 |\n|---|---|---|")
    for j, lab in enumerate(["Pred at t", "Update from t+1", "Update from t+3", "Update from t+5"]):
        print(f"| {lab} | {ms([o and o[j] for o in old])} | {ms([x and x[j] for x in dop])} |")
    print("\nAblation, ETH log-probability\n")
    arms = {"full": seeds,
            "w/o CRF": [f"configs_e43/s{i}/{FMC}/CIF" for i in (1, 2, 3)],
            "w/o CIF": [f"{FMC}/spline", f"configs_e43b/s2/{FMC}/spline", f"configs_e43b/s3/{FMC}/spline"]}
    paper = {"full": "-0.741 ± 0.104", "w/o CRF": "-0.799 ± 0.214", "w/o CIF": "-0.836 ± 0.257"}
    print("| Arm | paper (RK4) | dopri5 |\n|---|---|---|")
    for a, ds_ in arms.items():
        v = [vals(p(d, "eth", "thesis_dopri5_chain"), "log_prob_mean") for d in ds_]
        print(f"| {a} | {paper[a]} | {ms([x and x[0] for x in v])} |")
    print()


def ablation():
    print("## Ablation: Sinkhorn (mean ± std over 3 seeds)\n")
    print("paper | old GT now | exact GT now\n")
    arms = {"full": [f"{FMC}/CIF_spline", f"configs_seed/s2/{FMC}/CIF_spline", f"configs_seed/s3/{FMC}/CIF_spline"],
            "w/o CRF": [f"configs_e43/s{i}/{FMC}/CIF" for i in (1, 2, 3)],
            "w/o CIF": [f"{FMC}/spline", f"configs_e43b/s2/{FMC}/spline", f"configs_e43b/s3/{FMC}/spline"]}
    paper = {"full": ("1.387 ± 0.084", "1.279 ± 0.056"), "w/o CRF": ("1.464 ± 0.156", "1.333 ± 0.051"),
             "w/o CIF": ("1.253 ± 0.063", "1.344 ± 0.074")}
    print("| Arm | Fork | Branch |\n|---|---|---|")
    for a, ds_ in arms.items():
        cells = []
        for j, ds in enumerate(("simfork", "simbranch")):
            o = [vals(p(d, ds, "thesis_exactgt_abl"), "emd") for d in ds_]
            x = [vals(p(d, ds, "thesis_exactgt_abl"), "emd_exact") for d in ds_]
            cells.append(f"{paper[a][j]} \\| {ms([v and v[0] for v in o])} \\| {ms([v and v[0] for v in x])}")
        print(f"| {a} | {cells[0]} | {cells[1]} |")
    print()


def solver_table():
    print("## Solver table\n")
    rows = [("Dormand-Prince 5", "CIF_spline", 1.40, 0.93), ("Tsitouras 5", "CIF_spline_tsit5", 1.44, 0.97),
            ("Euler 1", "CIF_spline_euler_1", 2.00, 0.81), ("Euler 10", "CIF_spline_euler_10", 1.51, 0.91),
            ("Euler 100", "CIF_spline_euler_100", 1.43, 0.95), ("RK4 1", "CIF_spline_rk4_1", 1.77, 0.63),
            ("RK4 10", "CIF_spline_rk4_10", 1.35, 0.97), ("RK4 100", "CIF_spline_rk4_100", 1.40, 0.19)]
    print("| Solver | Fork: paper \\| old GT now \\| exact GT now | ETH log-prob: paper \\| now |\n|---|---|---|")
    for lab, v, pf, pl in rows:
        o = vals(p(f"{FMC}/{v}", "simfork", "thesis_exactgt"), "emd")
        x = vals(p(f"{FMC}/{v}", "simfork", "thesis_exactgt"), "emd_exact")
        e = vals(p(f"{FMC}/{v}", "eth", "thesis_solver"), "log_prob_mean")
        print(f"| {lab} | {pf:.2f} \\| {fmt(o and o[0])} \\| {fmt(x and x[0])} | {pl:.2f} \\| {fmt(e and e[0], 2)} |")
    print()


if __name__ == "__main__":
    main_table()
    eth_logprob()
    ablation()
    solver_table()
