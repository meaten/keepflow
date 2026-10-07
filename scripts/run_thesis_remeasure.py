"""Thesis re-measurement queue (no training, existing checkpoints only).

1. ETH log-probability with dopri5 (the paper's column was computed with RK4
   after the solver setting stopped being read on 2026-08-10).
2. Solver table, ETH column: RK4-100 (the paper's cell was a leftover first-pass
   result), with DOPRI5 and RK4-10 re-measured alongside as anchors.
3. Fork/Branch Sinkhorn against both the stored GT ("emd") and the exact
   posterior ("emd_exact", keepflow/data/traj/sim_gt.py) in the same run, for the
   main table, the ablation table and the solver table.

Each job writes metrics_<tag>.json next to its checkpoint; nothing existing is
overwritten. Finished jobs are skipped, so the queue is restartable.

    nohup pyenv exec python scripts/run_thesis_remeasure.py --devices 0 1 2 3 \
        > logs/thesis_rerun/driver.log 2>&1 &
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FMC = "configs/traj/FlowMatchingChain_onestep"
CNF_SEEDS = [f"{FMC}/CIF_spline",
             f"configs_seed/s2/{FMC}/CIF_spline",
             f"configs_seed/s3/{FMC}/CIF_spline"]
NOCRF = [f"configs_e43/s{i}/{FMC}/CIF" for i in (1, 2, 3)]
NOCIF = [f"{FMC}/spline", f"configs_e43b/s2/{FMC}/spline", f"configs_e43b/s3/{FMC}/spline"]
BASELINES = ["configs/traj/socialGAN", "configs/traj/Trajectron", "configs/traj/MID",
             "configs/traj/SingularTrajectory", "configs/traj/MoFlow/fm",
             "configs/traj/FlowChain/CIF_separate_cond_v"]
SOLVERS = ["CIF_spline", "CIF_spline_tsit5", "CIF_spline_euler_1", "CIF_spline_euler_10",
           "CIF_spline_euler_100", "CIF_spline_rk4_1", "CIF_spline_rk4_10", "CIF_spline_rk4_100"]

CHAIN = ["--scene_sampling", "per_agent", "--analytic_logprob", "--chain_logprob"]
FEB_SIM = ["--de_sample_size", "10", "--update_timesteps", "1", "3", "5"]       # run_configs.py --density --synthesized
FEB_ETH = ["--de_sample_size", "10", "--update_timesteps", "1", "2", "3", "4", "5", "6"]  # run_configs.py --density
ABL_SIM = ["--scene_sampling", "even", "--de_sample_size", "10", "--analytic_logprob",
           "--kde_n_sample", "10000"]


def jobs():
    J = []
    # 1. ETH log-prob, dopri5
    for d in CNF_SEEDS:
        J.append((f"{d}/eth.yml", CHAIN + ["--update_timesteps", "1", "3", "5"], "thesis_dopri5_chain"))
    for d in NOCRF + NOCIF:
        J.append((f"{d}/eth.yml", CHAIN, "thesis_dopri5_chain"))
    # 2. solver table, ETH column
    for v in ["CIF_spline", "CIF_spline_rk4_10", "CIF_spline_rk4_100"]:
        J.append((f"{FMC}/{v}/eth.yml", FEB_ETH, "thesis_solver"))
    # 3a. main table Fork/Branch
    for d in BASELINES + [CNF_SEEDS[0]]:
        for ds in ("simfork", "simbranch"):
            J.append((f"{d}/{ds}.yml", FEB_SIM, "thesis_exactgt"))
    # 3b. ablation Fork/Branch
    for d in CNF_SEEDS + NOCRF + NOCIF:
        for ds in ("simfork", "simbranch"):
            J.append((f"{d}/{ds}.yml", ABL_SIM, "thesis_exactgt_abl"))
    # 3c. solver table Fork column (CIF_spline/simfork is already in 3a)
    for v in SOLVERS[1:]:
        J.append((f"{FMC}/{v}/simfork.yml", FEB_SIM, "thesis_exactgt"))
    return J


def out_path(cfg, tag):
    return REPO / "experiments" / os.path.splitext(cfg)[0] / f"metrics_{tag}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--log_dir", default="logs/thesis_rerun")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    J = jobs()
    if args.status:
        done = sum(out_path(c, t).exists() for c, _, t in J)
        print(f"{done}/{len(J)} done")
        for c, _, t in J:
            print("  [done]" if out_path(c, t).exists() else "  [ -  ]", t, c)
        return
    q = queue.Queue()
    for j in J:
        if not out_path(j[0], j[2]).exists():
            q.put(j)
    print(f"{q.qsize()} jobs queued", flush=True)

    def worker(gpu):
        while True:
            try:
                cfg, extra, tag = q.get_nowait()
            except queue.Empty:
                return
            log = Path(args.log_dir) / f"{tag}__{cfg.replace('/', '_')}.log"
            cmd = [sys.executable, "test.py", "--config_file", cfg, "--device", f"cuda:{gpu}",
                   "--tag", tag] + extra
            t0 = time.time()
            with open(log, "w") as fp:
                rc = subprocess.call(cmd, cwd=REPO, stdout=fp, stderr=subprocess.STDOUT)
            print(f"[gpu{gpu}] rc={rc} {time.time() - t0:6.0f}s {tag} {cfg}", flush=True)

    ts = [threading.Thread(target=worker, args=(g,)) for g in args.devices]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
