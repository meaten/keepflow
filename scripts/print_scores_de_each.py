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

    fn = "metrics.json"
    
    for k, v in models.items():
        print(f"{k: <30}", end="\t")
        try:
            results = json.loads((dirpath / v / 'eth' / fn).read_text())
            value = results['log_prob_each_from_6'].strip('[]').split()
            value.append(results['log_prob_each_from_6_mean'])
            formatted_list = [f"{float(v):.2f}" for v in value]
            print("\t".join(formatted_list), end="\t")
            if k in ["FlowChain-NF", "FlowChain-CNF"]:
                print()
                print(" "*30, end="\t")
                value = results['log_prob_each_update_6_ahead'].strip('[]').split()
                value.append(results['log_prob_each_update_6_ahead_mean'])
                formatted_list = [f"{float(v):.2f}" for v in value]
                print("\t".join(formatted_list), end="\t")
        except KeyError:
            pass 
        print()

if __name__ == "__main__":
    main()