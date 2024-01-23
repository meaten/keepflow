import argparse
import itertools
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch training code for task-agnostic time-series prediction")
    parser.add_argument("--config_dir", type=str, default='',
                        metavar="DIR", help='path to the directory of config files')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--n_parallel", type=int, default=1)
    parser.add_argument("--test_only", action='store_true')
    return parser.parse_args()


def run_parallel(args: argparse.Namespace) -> None:
    if args.test_only:
        base_cmds = [["python3", "test.py"]]
    else:
        base_cmds = [["python3", "train.py", ], ["python3", "test.py"]]
    

    config_files = Path(args.config_dir).rglob("*.yml")
    
    cmds = []
    for base, c in itertools.product(base_cmds, config_files):
        cmd = base + [
                    "--device",
                    args.device,
                    "--config_file",
                    str(c)]
        cmds.append(cmd)
    
    for idx in range(0, len(cmds), args.n_parallel):
        cmds_para = cmds[idx:idx + args.n_parallel]
        [print(" ".join(cmd)) for cmd in cmds_para]
        procs = [subprocess.Popen(cmd) for cmd in cmds_para]
        [p.wait() for p in procs]


def main() -> None:
    args = parse_args()
    run_parallel(args)

    
if __name__ == "__main__":
    main()