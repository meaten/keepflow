"""Run a list of configs across several GPUs, one process per GPU at a time.

Each worker owns one GPU and pulls the next config off a shared queue, running
`train.py` then `test.py` for it. `train.py` resumes from `ckpt.pt` when one is
present, so the sweep is safe to interrupt and restart.

Designed to be launched detached, e.g.

    nohup python scripts/run_seed_sweep.py \
        --config_glob 'configs_seed/*/configs/traj/**/*.yml' \
        --devices 0 1 2 --log_dir logs/seed_sweep > logs/seed_sweep/driver.log 2>&1 &

Progress can be checked at any time with --status.
"""
import argparse
import glob
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_glob", required=True,
                   help="glob for the config files to run (quote it)")
    p.add_argument("--devices", nargs="+", type=int, default=[0],
                   help="GPU indices; one worker process per index")
    p.add_argument("--log_dir", default="logs/seed_sweep")
    p.add_argument("--save_dir", default="experiments",
                   help="must match SAVE_DIR; used to detect finished runs")
    p.add_argument("--skip_done", action="store_true",
                   help="skip configs whose metrics.json already exists")
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--density", action="store_true",
                   help="run the density-estimation pass (EMD / log-probability / timing) "
                        "instead of the plain ADE/FDE test; implies --test_only and writes "
                        "metrics_<density_tag>.json")
    p.add_argument("--density_tag", default="density")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--status", action="store_true",
                   help="print completion status for the glob and exit")
    return p.parse_args()


def out_dir(save_dir: Path, cfg: Path) -> Path:
    return save_dir / cfg.with_suffix("")


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def worker(dev: int, q: "queue.Queue[Path]", log_dir: Path,
           test_only: bool, dry_run: bool, failures: list,
           density: bool = False, density_tag: str = "density") -> None:
    while True:
        try:
            cfg = q.get_nowait()
        except queue.Empty:
            return

        stages = ["test.py"] if (test_only or density) else ["train.py", "test.py"]
        tag = str(cfg.with_suffix("")).replace("/", "__")
        for stage in stages:
            cmd = [PYTHON, stage, "--device", f"cuda:{dev}",
                   "--config_file", str(cfg)]
            if density and stage == "test.py":
                # same protocol as the paper's Table 1 / Table 2
                cmd += ["--de_sample_size", "10", "--kde_n_sample", "10000",
                        "--update_timesteps", "1", "3", "5", "--tag", density_tag]
            log(f"GPU{dev} START {stage} {cfg}")
            if dry_run:
                log(f"GPU{dev} (dry-run) {' '.join(cmd)}")
                continue
            logfile = log_dir / f"{tag}.{stage.replace('.py', '')}.log"
            t0 = time.time()
            with open(logfile, "w") as fp:
                rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=fp,
                                     stderr=subprocess.STDOUT)
            dt = (time.time() - t0) / 3600
            if rc != 0:
                log(f"GPU{dev} FAIL  {stage} {cfg} (rc={rc}, {dt:.2f}h) -> {logfile}")
                failures.append((str(cfg), stage, rc))
                break  # do not test a model that failed to train
            log(f"GPU{dev} DONE  {stage} {cfg} ({dt:.2f}h)")
        q.task_done()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    configs = sorted(Path(p) for p in glob.glob(args.config_glob, recursive=True))

    if args.status:
        done = [c for c in configs if (out_dir(save_dir, c) / "metrics.json").exists()]
        ckpt = [c for c in configs if (out_dir(save_dir, c) / "ckpt.pt").exists()]
        print(f"{len(configs)} configs | {len(ckpt)} with ckpt | {len(done)} with metrics")
        for c in configs:
            d = out_dir(save_dir, c)
            mark = ("done " if (d / "metrics.json").exists()
                    else "train" if (d / "ckpt.pt").exists() else "  -  ")
            print(f"  [{mark}] {c}")
        return

    if not configs:
        sys.exit(f"no configs matched: {args.config_glob}")

    if args.skip_done:
        before = len(configs)
        marker = f"metrics_{args.density_tag}.json" if args.density else "metrics.json"
        configs = [c for c in configs
                   if not (out_dir(save_dir, c) / marker).exists()]
        log(f"skipping {before - len(configs)} already-finished config(s)")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log(f"{len(configs)} config(s) over GPUs {args.devices}")
    q: "queue.Queue[Path]" = queue.Queue()
    for c in configs:
        q.put(c)

    failures: list = []
    threads = [threading.Thread(target=worker,
                                args=(d, q, log_dir, args.test_only,
                                      args.dry_run, failures,
                                      args.density, args.density_tag), daemon=False)
               for d in args.devices]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log(f"sweep finished in {(time.time() - t0) / 3600:.2f}h; "
        f"{len(failures)} failure(s)")
    for cfg, stage, rc in failures:
        log(f"  FAILED {stage} {cfg} rc={rc}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
