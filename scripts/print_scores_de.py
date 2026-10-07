import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="experiments/configs/traj/")
    return parser.parse_args()

def main():
    args = parse_args()
    dirpath = Path(args.dir)

    models = {
        "Social-GAN": "socialGAN",
        "Trajectron++": "Trajectron",
        "MID": "MID",
        "SingularTrajectory": "SingularTrajectory",
        "MoFlow": "MoFlow/fm",
        "FlowChain-NF": "FlowChain/CIF_separate_cond_v",
        "FlowChain-CNF": "FlowMatchingChain_onestep/CIF_spline"
    }
    
    datasets = ['simunicycle', 'simfork', 'simbranch', "eth"]
    fn = "metrics.json"
    
    for k, v in models.items():
        print(f"{k: <30}", end="\t")
        for dn in datasets:
            metric = 'emd' if 'sim' in dn else 'log_prob_mean'
            try:
                value = json.loads((dirpath / v / dn / fn).read_text())[metric].strip('[]').split()[0]
                print(f"{dn}: {float(value):.2f}", end="\t")
            except KeyError:
                pass

        if k in ["FlowChain-NF", "FlowChain-CNF"]:
            steps = [1, 2, 3]
            for s in steps:
                print()
                print(f"update t+{s: <21}", end="\t")
                for dn in datasets:
                    metric = 'emd' if 'sim' in dn else 'log_prob_mean'
                    step = s if 'sim' in dn else {1: 1, 2:3, 3:5}[s]
                    try:
                        value = json.loads((dirpath / v / dn / fn).read_text())[metric].strip('[]').split()[step]
                        print(f"{dn}: {float(value):.2f}", end="\t")
                    except KeyError:
                        pass
        print()


if __name__ == "__main__":
    main()