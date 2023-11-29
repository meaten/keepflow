import os
import argparse
from typing import List, Dict
from yacs.config import CfgNode
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm

from keepflow.utils import load_config, Timer, GaussianKDE
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics
from keepflow.visualization import build_visualizer

from train import evaluate_model, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch testing code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--visualize", action="store_true", help="flag for visualizing results")
    
    return parser.parse_args()


def test(cfg: CfgNode, flag_visualize: bool=False) -> None:
    data_loader = build_dataloader(cfg, rand=False, split="test")
    model = build_model(cfg)
    try:
        model.load()
    except FileNotFoundError:
        print("no model saved")
        
    result_metrics = {}
    
    if flag_visualize:
        result_metrics.update(visualize(cfg, model))
    result_metrics.update(evaluate_model(cfg, model, data_loader))
    np.set_printoptions(precision=4)
    print(result_metrics)
    import json
    with open(os.path.join(cfg.SAVE_DIR, "metrics.json"), "w") as fp:
        json.dump(result_metrics, fp)


def visualize(cfg, model: torch.nn.Module):
    metrics = build_metrics(cfg)
    model.eval()
    
    timer = Timer(cfg.DEVICE)
    
    update_timesteps = [1, 2, 3, 4, 5, 6]
    update_timesteps = [1]
    
    result_metrics = {}
    
    run_times = {0: []}
    run_times.update({t: [] for t in update_timesteps})

    visualizer = build_visualizer(cfg)
    with torch.no_grad():
        result_list = []
        print("timing the computation, evaluating probability map, and visualizing... ")
        data_loaders = build_dataloader(cfg, rand=False, split = "test", batch_size=1)
        for node_type, dataloader in data_loaders.items():
            n_vis = 5
            for i, data_dict in enumerate(tqdm(dataloader, total=n_vis)):
                data_dict = {k: data_dict[k].to(cfg.DEVICE) 
                            if isinstance(data_dict[k], torch.Tensor)
                            else data_dict[k]
                            for k in data_dict}
                dict_list = []
                
                result_dict = model.predict(deepcopy(data_dict), return_prob=True)  # warm-up
                timer.start()
                result_dict = model.predict(deepcopy(data_dict), return_prob=True)
                timer.end()
                curr_run_time = timer.elapsed_time()
                run_times[0].append(curr_run_time)
                    
                for t in update_timesteps:
                    timer.start()
                    result_dict = model.predict_from_new_obs(result_dict, t)
                    timer.end()
                    curr_run_time = timer.elapsed_time()
                    run_times[t].append(curr_run_time)
                    
                dict_list.append(deepcopy(result_dict))
                dict_list = metrics.denormalize(dict_list)  # denormalize the output
                if cfg.TEST.KDE:
                    timer.start()
                    dict_list = kde(dict_list)
                    timer.end()
                    run_times[0][-1] += timer.elapsed_time()
                timer.start()
                dict_list = visualizer.prob_on_grid(dict_list)
                timer.end()
                run_times[0][-1] += timer.elapsed_time()
                result_list.append(metrics(deepcopy(dict_list)))
                
                if visualize:
                    visualizer(dict_list)
                if i == n_vis - 1:
                    break
                
        result_metrics.update(aggregate(result_list))
        
    print(f"execution time: {np.mean(run_times[0]):.2f} " + u"\u00B1" + f"{np.std(run_times[0]):.2f} [ms]")
    print(f"execution time: {np.mean(run_times[1]):.2f} " + u"\u00B1" + f"{np.std(run_times[1]):.2f} [ms]")
    result_metrics.update({"execution time": np.mean(run_times[0]), "time std": np.std(run_times[0])})
    
    # from visualization.TP_visualizer import plot2d_trajectories_samples
    # import dill
    # from pathlib import Path
    # env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "processed_data" / f"{cfg.DATA.DATASET_NAME}_test.pkl"
    # with open(env_path, 'rb') as f:
    #     env = dill.load(f, encoding='latin1')
    # max_pos, min_pos = env.scenes[0].calculate_pos_min_max()
    # max_pos += 0.05 * (max_pos - min_pos)
    # min_pos -= 0.05 * (max_pos - min_pos)
    # gts = model.gts
    # obss = model.obss
    # plot2d_trajectories_samples(obss, gts, max_pos, min_pos)
    # import pdb;pdb.set_trace()
    
    model.train()
    
    return result_metrics
        
        
def kde(dict_list: List):
    for data_dict in dict_list:
        for k in list(data_dict.keys()):
            if k[0] == "prob":
                prob = data_dict[k]
                batch_size, _, timesteps, _ = prob.shape
                prob_, gt_traj_log_prob = [], []
                for b in range(batch_size):
                    prob__, gt_traj_prob__ = [], []
                    for i in range(timesteps):
                        kernel = GaussianKDE(prob[b, :, i, :-1])
                        kernel(prob[b, :, i, :-1])  # estimate the prob of predicted future positions for fair comparison of inference time
                        prob__.append(deepcopy(kernel))
                        gt_traj_prob__.append(kernel(data_dict["gt"][b, None, i].float()))
                    prob_.append(deepcopy(prob__))
                    gt_traj_log_prob.append(torch.cat(gt_traj_prob__, dim=-1).log())
                gt_traj_log_prob = torch.stack(gt_traj_log_prob, dim=0)
                gt_traj_log_prob = torch.nan_to_num(gt_traj_log_prob, neginf=-10000)
                data_dict[k] = prob_
                data_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob
            
    return dict_list


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    test(cfg, args.visualize)

if __name__ == "__main__":
    main()