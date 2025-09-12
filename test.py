import os
import argparse
from yacs.config import CfgNode
import json
import numpy as np
import torch
from copy import deepcopy

from keepflow.utils import load_config, Timer, kde
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics
from keepflow.visualization import build_visualizer

from train import evaluate_model, aggregate, dict_to_device


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    test(cfg, num_visualize=args.visualize)
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch testing code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--visualize", type=int, default=0, help="number of samples to visualize")
    
    return parser.parse_args()


def test(cfg: CfgNode, num_visualize: int=0) -> None:
    data_loader = build_dataloader(cfg, rand=False, split="test")
    model = build_model(cfg)
    try:
        model.load()
    except FileNotFoundError:
        print("no model saved")
    
    result_metrics = {}

    if num_visualize > 0:
        result_metrics.update(visualize(cfg, model, num_visualize))
    result_metrics.update(evaluate_model(cfg, model, data_loader))
    np.set_printoptions(precision=4)
    print(result_metrics)
    result_metrics = {k: str(v) for k, v in result_metrics.items()}
    with open(os.path.join(cfg.SAVE_DIR, "metrics.json"), "w") as fp:
        json.dump(result_metrics, fp)


def visualize(cfg, model: torch.nn.Module, num_visualize: int) -> dict:
    metrics = build_metrics(cfg)
    model.eval()
    
    timer = Timer(cfg.DEVICE)
    
    update_timesteps = [1, 2]
    # update_timesteps = []
    
    result_metrics = {}
    
    run_times = {0: []}
    run_times.update({t: [] for t in update_timesteps})
    
    visualizer = build_visualizer(cfg)
    with torch.no_grad():
        result_list = []
        print("timing the computation, evaluating probability map, and visualizing... ")
        data_loader = build_dataloader(cfg, rand=False, split = "test", batch_size=1)
        prev_idx = None
        count = 0
        for data_dict in data_loader:
            data_dict = dict_to_device(data_dict, cfg.DEVICE)
            
            # if data_dict["index"][0][2] != '2_6' or data_dict["index"][0][1].item() != 14:
            if data_dict["index"][0][2] == prev_idx:
                continue
            else:
                prev_idx = data_dict["index"][0][2]  # agent_id
                count += 1
                print(f"visualize {count} / {num_visualize}")

            result_dict = model.predict(data_dict, return_prob=True)  # warm-up
            timer.start()
            result_dict = model.predict(data_dict, return_prob=True)
            timer.end()
            curr_run_time = timer.elapsed_time()
            run_times[0].append(curr_run_time)
            
            for t in update_timesteps:
                timer.start()
                result_dict = model.update_from_new_obs(result_dict, t)
                timer.end()
                curr_run_time = timer.elapsed_time()
                run_times[t].append(curr_run_time)
            
            result_dict = metrics.denormalize(result_dict)  # denormalize the output
            if cfg.TEST.KDE:
                timer.start()
                result_dict = kde(result_dict)
                timer.end()
                run_times[0][-1] += timer.elapsed_time()
            result_dict = visualizer.prob_on_grid(result_dict)
            result_list.append(metrics(result_dict))
            
            visualizer(result_dict)
            if count == num_visualize:
                break
            
        result_metrics.update(aggregate(result_list))
        
    print(f"execution time: {np.mean(run_times[0]):.2f} " + u"\u00B1" + f"{np.std(run_times[0]):.2f} [ms]")
    if len(update_timesteps) != 0:
        print(f"execution time: {np.mean(run_times[1]):.2f} " + u"\u00B1" + f"{np.std(run_times[1]):.2f} [ms]")
    result_metrics.update({"execution time": np.mean(run_times[0]), "time std": np.std(run_times[0])})
    
    model.train()
    
    return result_metrics

if __name__ == "__main__":
    main()