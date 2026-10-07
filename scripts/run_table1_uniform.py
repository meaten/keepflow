"""Re-measure the paper's Table 1 under one uniform protocol.

The published Table 1 mixes numbers carried over from the authors' ICCV 2023
paper (measured on different hardware, with the pre-vectorization loop-based
KDE) with numbers measured freshly. For a fair speed comparison every cell is
re-measured here on the same GPU with the current code.

Runs `test.py` once per (method, dataset), writing `metrics_uniform.json` next
to each checkpoint so nothing existing is overwritten. Combinations that
already have that file are skipped, so the sweep is restartable.

    nohup pyenv exec python scripts/run_table1_uniform.py --device 3 \
        > logs/table1_uniform/driver.log 2>&1 &
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (label, config directory)
METHODS = [
    ("Social-GAN",         "configs/traj/socialGAN"),
    ("Trajectron++",       "configs/traj/Trajectron"),
    ("MID",                "configs/traj/MID"),
    ("SingularTrajectory", "configs/traj/SingularTrajectory"),
    ("MoFlow",             "configs/traj/MoFlow/fm"),
    ("FlowChain-NF",       "configs/traj/FlowChain/CIF_separate_cond_v"),
    ("FlowChain-CNF",      "configs/traj/FlowMatchingChain_onestep/CIF_spline"),
]
DATASETS = ["eth", "simunicycle", "simfork", "simbranch"]

TAG = "uniform"
DE_SAMPLE_SIZE = 10      # paper protocol (yields 9 scenes, see plan §1 #19)
KDE_N_SAMPLE = 10000     # paper protocol
UPDATE_TIMESTEPS = ["1", "3", "5"]


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=3,
                   help="GPU index; keep off the GPUs running the training sweep")
    p.add_argument("--save_dir", default="experiments")
    p.add_argument("--log_dir", default="logs/table1_uniform")
    p.add_argument("--wait_for_free", action="store_true",
                   help="wait until no other test.py --tag uniform process is running")
    p.add_argument("--status", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def out_path(save_dir, cfg_dir, ds):
    return Path(save_dir) / cfg_dir / ds / f"metrics_{TAG}.json"


def main():
    args = parse_args()
    combos = [(label, cfg_dir, ds) for label, cfg_dir in METHODS for ds in DATASETS]

    if args.status:
        done = 0
        for label, cfg_dir, ds in combos:
            p = out_path(args.save_dir, cfg_dir, ds)
            mark = "done" if p.exists() else " -  "
            done += p.exists()
            print(f"  [{mark}] {label:20s} {ds}")
        print(f"{done}/{len(combos)} complete")
        return

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.wait_for_free:
        while subprocess.call(["pgrep", "-f", f"test.py.*--tag {TAG}"],
                              stdout=subprocess.DEVNULL) == 0:
            log("another uniform run is in flight; waiting 60 s")
            time.sleep(60)

    todo, skipped, missing = [], 0, []
    for label, cfg_dir, ds in combos:
        cfg = Path(cfg_dir) / f"{ds}.yml"
        if not (REPO_ROOT / cfg).exists():
            missing.append(f"{label}/{ds}: no config {cfg}")
            continue
        if not (REPO_ROOT / args.save_dir / cfg_dir / ds / "ckpt.pt").exists():
            missing.append(f"{label}/{ds}: no checkpoint")
            continue
        if out_path(args.save_dir, cfg_dir, ds).exists():
            skipped += 1
            continue
        todo.append((label, cfg_dir, ds, cfg))

    log(f"{len(todo)} to run, {skipped} already done, {len(missing)} unavailable")
    for m in missing:
        log(f"  UNAVAILABLE {m}")

    failures = []
    t_start = time.time()
    for i, (label, cfg_dir, ds, cfg) in enumerate(todo, 1):
        cmd = [sys.executable, "test.py",
               "--config_file", str(cfg),
               "--device", f"cuda:{args.device}",
               "--de_sample_size", str(DE_SAMPLE_SIZE),
               "--kde_n_sample", str(KDE_N_SAMPLE),
               "--update_timesteps", *UPDATE_TIMESTEPS,
               "--tag", TAG]
        log(f"({i}/{len(todo)}) {label} / {ds}")
        if args.dry_run:
            log("  " + " ".join(cmd))
            continue
        logfile = log_dir / f"{label.replace('/', '_')}__{ds}.log"
        t0 = time.time()
        with open(logfile, "w") as fp:
            rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=fp,
                                 stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if rc != 0:
            log(f"  FAIL rc={rc} after {dt/60:.1f} min -> {logfile}")
            failures.append((label, ds, rc))
        else:
            log(f"  ok ({dt/60:.1f} min)")

    log(f"finished in {(time.time()-t_start)/60:.1f} min; {len(failures)} failure(s)")
    for label, ds, rc in failures:
        log(f"  FAILED {label}/{ds} rc={rc}")


if __name__ == "__main__":
    main()
