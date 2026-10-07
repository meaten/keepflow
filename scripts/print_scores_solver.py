import argparse
import json
from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="experiments/configs/traj/")
    return parser.parse_args()

def main():
    args = parse_args()
    dirpath = Path(args.dir)

    models = {
        "Dormant-Prince 5": "FlowMatchingChain_onestep/CIF_spline",
        "Tsitouras 5": "FlowMatchingChain_onestep/CIF_spline_tsit5",
        "Euler 1": "FlowMatchingChain_onestep/CIF_spline_euler_1",
        "Euler 10": "FlowMatchingChain_onestep/CIF_spline_euler_10",
        "Euler 100": "FlowMatchingChain_onestep/CIF_spline_euler_100",
        "rk4 1": "FlowMatchingChain_onestep/CIF_spline_rk4_1",
        "rk4 10": "FlowMatchingChain_onestep/CIF_spline_rk4_10",
        "rk4 100": "FlowMatchingChain_onestep/CIF_spline_rk4_100",
    }

    datasets = ['simfork', 'eth']
    fn = "metrics.json"
    
    for k, v in models.items():
        print(f"{k: <40}", end="\t")
        value = json.loads((dirpath / v / 'simfork' / fn).read_text())['execution time']
        print(f"{float(value):.2f}", end="\t")
        for dn in datasets:
            metric = 'emd' if 'sim' in dn else 'log_prob_mean'
            try:
                value = json.loads((dirpath / v / dn / fn).read_text())[metric].strip('[]').split()[0]
                print(f"{float(value):.2f}", end="\t")
            except KeyError:
                pass
        print()

if __name__ == "__main__":
    main()