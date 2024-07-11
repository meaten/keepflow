import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="experiments/configs/traj/FlowMatchingChain/CIF_separate_cond_test/")
    return parser.parse_args()

def main():
    args = parse_args()
    dirpath = Path(args.dir)
    
    dataset = "*"
    fn = "metrics.json"
    
    metrics = ["score", "ade", "fde", "log_prob"]
    # metrics = ["ade", "fde"]
    
    matches = [p for p in dirpath.glob("/".join([dataset, fn]))]
    matches.sort(key=lambda x: x.parent.name)
    
    score_dir = {}
    for match in matches:
        score_dir[match.parent.name] = json.loads(match.read_text())
    
    for k in score_dir:
        print(f"{k: <50}", end="\t")
        for metric in metrics:
            try:
                print(f"{metric}: {score_dir[k][metric]}", end="\t")
            except KeyError:
                pass
        print()


if __name__ == "__main__":
    main()