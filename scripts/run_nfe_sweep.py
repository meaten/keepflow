"""Accuracy-vs-NFE sweep: does CRF (spline target) buy step-efficiency?

Rectified flow's claimed benefit is not a better final distribution but a
straighter probability path, so that the ODE can be integrated with fewer
function evaluations. The de-confounded ablation compared the two arms at the
*full* solver budget (adaptive dopri5, tol 1e-2) and found parity -- which is
exactly what that theory predicts. This script measures the axis where a
difference should appear instead: accuracy as a function of the integration
budget.

Both arms are evaluated from their existing checkpoints; only the inference
solver is overridden (`test.py --cnf_solver/--num_step`), so no retraining is
involved and the comparison is exactly like-for-like.

  CRF arm     : configs/traj/.../CIF_spline (seed 1) + configs_seed/s{2,3}
  no-CRF arm  : configs_e43/s{1,2,3}/.../CIF

Results land in metrics_<tag>.json next to each checkpoint, tag = nfe_<setting>.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/maeda/.pyenv/versions/keepflow/bin/python"

DATASETS = ["eth", "simfork", "simbranch", "simunicycle"]

# (arm, seed) -> directory holding the per-dataset yml files
ARMS = {
    "crf": {
        1: "configs/traj/FlowMatchingChain_onestep/CIF_spline",
        2: "configs_seed/s2/configs/traj/FlowMatchingChain_onestep/CIF_spline",
        3: "configs_seed/s3/configs/traj/FlowMatchingChain_onestep/CIF_spline",
    },
    "nocrf": {
        1: "configs_e43/s1/configs/traj/FlowMatchingChain_onestep/CIF",
        2: "configs_e43/s2/configs/traj/FlowMatchingChain_onestep/CIF",
        3: "configs_e43/s3/configs/traj/FlowMatchingChain_onestep/CIF",
    },
}

# (tag, solver, num_step). num_step=None -> adaptive solver, no override of the
# step count. "full" is the paper's default setting and is run without any
# override at all so it also serves as a regression check.
SETTINGS = [
    ("euler1", "euler", 1),
    ("euler2", "euler", 2),
    ("euler3", "euler", 3),
    ("euler5", "euler", 5),
    ("euler10", "euler", 10),
    ("rk41", "rk4", 1),
    ("rk42", "rk4", 2),
    ("full", None, None),
]


def build_jobs(settings, arms, datasets, seeds, density):
    jobs = []
    for arm, seed, ds, (tag, solver, nstep) in itertools.product(
            arms, seeds, datasets, settings):
        # The config path MUST stay relative: load_config builds SAVE_DIR as
        # os.path.join("experiments", <config path minus extension>), and
        # os.path.join drops the "experiments" prefix outright when handed an
        # absolute path -- which silently redirects results next to the yml and
        # makes model.load() miss the checkpoint ("no model saved"), i.e. an
        # untrained model gets evaluated. Subprocesses run with cwd=REPO.
        cfg = os.path.join(ARMS[arm][seed], f"{ds}.yml")
        if not os.path.exists(os.path.join(REPO, cfg)):
            print(f"[skip] missing config {cfg}", file=sys.stderr)
            continue
        cmd = [PY, "test.py", "--config_file", cfg, "--tag", f"nfe_{tag}"]
        if solver is not None:
            cmd += ["--cnf_solver", solver]
            if nstep is not None:
                cmd += ["--num_step", str(nstep)]
        if density:
            cmd += ["--de_sample_size", "10", "--kde_n_sample", "10000"]
        jobs.append((f"{arm}-s{seed}-{ds}-{tag}", cmd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=["crf", "nocrf"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--settings", nargs="+", default=None,
                    help="subset of setting tags, e.g. euler1 euler5 full")
    ap.add_argument("--density", action="store_true",
                    help="also compute EMD/log-prob (much slower)")
    ap.add_argument("--log_dir", default=os.path.join(REPO, "logs", "nfe_sweep"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    settings = SETTINGS
    if args.settings is not None:
        settings = [s for s in SETTINGS if s[0] in args.settings]

    jobs = build_jobs(settings, args.arms, args.datasets, args.seeds, args.density)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"{len(jobs)} runs over GPUs {args.gpus}")
    if args.dry_run:
        for name, cmd in jobs:
            print(name, " ".join(cmd))
        return

    slots = list(args.gpus)
    t0 = time.time()
    done = [0]

    def run(idx_job):
        idx, (name, cmd) = idx_job
        gpu = slots[idx % len(slots)]
        full = cmd + ["--device", f"cuda:{gpu}"]
        log = os.path.join(args.log_dir, f"{name}.log")
        with open(log, "w") as fp:
            rc = subprocess.call(full, cwd=REPO, stdout=fp, stderr=subprocess.STDOUT)
        # test.py swallows a missing checkpoint and happily evaluates an
        # untrained model, so a zero exit code is not enough to trust the run.
        if rc == 0 and "no model saved" in open(log, errors="ignore").read():
            rc = "NO_CKPT"
        done[0] += 1
        print(f"[{done[0]}/{len(jobs)}] {name} rc={rc} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
        return name, rc

    with ThreadPoolExecutor(max_workers=len(slots)) as ex:
        results = list(ex.map(run, enumerate(jobs)))

    failed = [n for n, rc in results if rc != 0]
    print(f"\nDONE in {time.time() - t0:.0f}s. {len(failed)} failures.")
    for n in failed:
        print("  FAILED", n)


if __name__ == "__main__":
    main()
