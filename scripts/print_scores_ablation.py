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
        "FlowChain-NF": "FlowChain/CIF_separate_cond_v",
        "FlowChain-NF (w/o CIF)": "FlowChain/separate_cond_v",
        "FlowChain-CNF": "FlowMatchingChain_onestep/CIF_spline",
        "FlowChain-CNF (w/o CIF)": "FlowMatchingChain_onestep/spline",
        "FlowChain-CNF (w/o CIFM)": "ContinuousFlowChain_onestep/CIF",
        "FlowChain-CNF (w/o Consistent Rectified Flow)": "FlowMatchingChain_onestep/CIF"
    }

    datasets = ['simfork', 'simbranch', 'simunicycle', 'eth']
    fn = "metrics.json"
    
    for k, v in models.items():
        print(f"{k: <40}", end="\t")
        if k == 'FlowChain-CNF (w/o CIFM)':
            print()
            continue
        for dn in datasets:
            metric = 'emd' if 'sim' in dn else 'log_prob_mean'
            try:
                value = json.loads((dirpath / v / dn / fn).read_text())[metric].strip('[]').split()[0]
                # print(f"{dn}: {float(value):.2f}", end="\t")
                print(f"{float(value):.2f}", end="\t")
            except KeyError:
                pass
        pred_metrics = {'ade': [], 'fde': []}
        for dn in ['eth', 'univ', 'hotel', 'zara1', 'zara2']:
            results = json.loads((dirpath / v / dn / fn).read_text())
            pred_metrics['ade'].append(float(results['ade']))
            pred_metrics['fde'].append(float(results['fde']))
        for k, v in pred_metrics.items():
            # print(f"{k}: {np.mean(v):.2f}", end='\t')
            print(f"{np.mean(v):.2f}", end='\t')
        print()

if __name__ == "__main__":
    main()