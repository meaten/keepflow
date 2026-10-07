import argparse
import itertools
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch training code for task-agnostic time-series prediction")
    parser.add_argument("--config_dir", type=str,
                        metavar="DIR", help='path to the directory of config files')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--n_parallel", type=int, default=1)
    parser.add_argument("--test_only", action='store_true')
    parser.add_argument("--visualize_only", action='store_true')
    parser.add_argument("--density", action='store_true')
    parser.add_argument("--dry_run", action='store_true')
    parser.add_argument("--synthesized", action='store_true')
    return parser.parse_args()


def run_parallel(args: argparse.Namespace) -> None:
    train_cmd = ["python3", "train.py", ]
    test_cmd = ["python3", "test.py"]
    visualize_cmd = ["python3", "visualize.py"]
    if args.density:
        if args.synthesized:
            test_cmd.extend(["--de_sample_size", "10", "--update_timesteps", "1", "3", "5"])
        else:
            # test_cmd.extend(["--de_sample_size", "-1", "--update_timesteps", "1", "2", "3", "4", "5", "6"])
            test_cmd.extend(["--de_sample_size", "10", "--update_timesteps", "1", "2", "3", "4", "5", "6"])

    if args.test_only:
        base_cmds = [test_cmd]
    elif args.visualize_only:
        base_cmds = [visualize_cmd]
    else:
        base_cmds = [train_cmd, test_cmd]
    

    if args.visualize_only:
        targets = ['simunicycle', 'simfork', 'simbranch', 'eth', 'zara1']
        config_files = [Path(args.config_dir) / (t + '.yml')  for t in targets]
    elif args.synthesized:
        config_files = Path(args.config_dir).rglob("sim*.yml")
    else:
        if not args.density and not args.visualize_only:
            targets = ['eth', 'univ', 'hotel', 'zara1', 'zara2']
        elif args.density:
            targets = ['eth']
        config_files = [Path(args.config_dir) / (t + '.yml')  for t in targets]
        config_files = [p for p in config_files if p.exists()]
    
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
        if not args.dry_run:
            procs = [subprocess.Popen(cmd) for cmd in cmds_para]
            [p.wait() for p in procs]


def main() -> None:
    args = parse_args()
    run_parallel(args)

    
if __name__ == "__main__":
    main()