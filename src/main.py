import argparse
from typing import List, Dict
from yacs.config import CfgNode
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict

from utils import load_config
from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics
from visualization.build_visualizer import Build_Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch training code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "tune"], default="train")
    parser.add_argument(
        "--visualize", action="store_true")
    
    return parser.parse_args()


def train(cfg: CfgNode) -> None:
    if cfg.USE_WANDB:
        import wandb
        wandb.init(project=f"{cfg.DATA.TASK}_{cfg.DATA.DATASET_NAME}", name=cfg.MODEL.TYPE)

    data_loader = unified_loader(cfg, rand=True, split="train")
    if cfg.DATA.TASK in ["MP", "TP"]:
        val_data_loader = unified_loader(cfg, rand=False, split="val")
        
    try:
        model = Build_Model(cfg, load=cfg.SOLVER.TRY_LOAD)
        print('loaded pretrained model')
    except FileNotFoundError:
        model = Build_Model(cfg)
    model.cuda()

    if cfg.SOLVER.USE_SCHEDULER:
        schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.SOLVER.ITER/10), gamma=0.80) for optimizer in model.optimizers]
    
    val_loss = np.inf
    with tqdm(range(cfg.SOLVER.ITER)) as pbar:
        for i in pbar:
            loss_list = []
            for data_dict in data_loader:
                data_dict = {k: data_dict[k].cuda() for k in data_dict}
                
                loss_list.append(model.update(data_dict))

            loss_info = aggregate(loss_list)
            pbar.set_postfix(OrderedDict(loss_info))
            
            if cfg.USE_WANDB:
                wandb.log(loss_info, step=i)

            # validation
            if (i+1) % cfg.SOLVER.SAVE_EVERY == 0:
                if cfg.DATA.TASK in ["VP"] or not cfg.SOLVER.VALIDATION:
                    model.save()
                else:
                    curr_val_loss = evaluate_model(cfg, model, val_data_loader)["score"]
                    if curr_val_loss < val_loss:
                        model.save()
                        val_loss = curr_val_loss

            
        if cfg.SOLVER.USE_SCHEDULER:
            [scheduler.step() for scheduler in schedulers]


def evaluate_model(cfg: CfgNode, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, visualize=False):
    model.eval()
    metrics = Build_Metrics(cfg)
    visualizer = Build_Visualizer(cfg)
    
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    run_times = []
    with torch.no_grad():
        result_list = []
        for data_dict in tqdm(data_loader):
            data_dict = {k: data_dict[k].cuda() for k in data_dict}
            
            dict_list = []
            torch.cuda.synchronize()
            for _ in range(cfg.TEST.N_TRIAL):
                starter.record()
                dict_list.append(deepcopy(model.predict(data_dict)))
                ender.record()
                torch.cuda.synchronize()
                curr_run_time = starter.elapsed_time(ender)
                run_times.append(curr_run_time)
                
                # release GPU memory
                if "prob" in dict_list[-1]:
                    dict_list[-1]["prob"] = dict_list[-1]["prob"].cpu()
                
            result_list.append(deepcopy(metrics(dict_list)))
            if visualize:
                visualizer(dict_list)
            
    result_info = aggregate(result_list)

    np.set_printoptions(precision=4)
    print(result_info)
    print(f"execution time: {np.mean(run_times):.2f} " + u"\u00B1" + f"{np.std(run_times):.2f} [ms]")
    model.train()

    return result_info
            

def test(cfg: CfgNode, visualize) -> None:
    data_loader = unified_loader(cfg, rand=False, split="test")
    model = Build_Model(cfg, load=True)
    evaluate_model(cfg, model, data_loader, visualize)

def aggregate(dict_list: List[Dict]) -> Dict:
    if "nsample" in dict_list[0]:
        ret_dict = {k: np.sum([d[k] for d in dict_list], axis=0) / np.sum([d["nsample"] for d in dict_list]) for k in dict_list[0].keys()}
    else:
        ret_dict = {k: np.mean([d[k] for d in dict_list], axis=0) for k in dict_list[0].keys()}

    return ret_dict
    
    
def tune(cfg: CfgNode) -> None:
    pass


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "test":
        test(cfg, args.visualize)
    elif args.mode == "tune":
        tune(cfg)

if __name__ == "__main__":
    main()