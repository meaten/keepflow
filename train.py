import argparse
from typing import List, Dict
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
import dill
from yacs.config import CfgNode
import numpy as np
import torch

from keepflow.utils import load_config
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    train(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch training code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--device", type=str, default="cuda:0")
    
    return parser.parse_args()


def train(cfg: CfgNode, save_model=True, load_model=True) -> None:        
    validation = cfg.SOLVER.VALIDATION and cfg.DATA.TASK != "video" 
    
    data_loader = build_dataloader(cfg, rand=True, split="train")
    # we don't have any validation set for Video Prediction
    if validation:
        val_data_loader = build_dataloader(cfg, rand=False, split="val")
        val_score = np.inf
    
    start_epoch = 0
    model = build_model(cfg)
    
    if model.check_saved_path() and load_model:
        # model weights are saved at the end of each epoch.
        # If model weights are saved, load and resume training from next epoch
        start_epoch = model.load() + 1
        print('loaded trained model, and resume training')

    if cfg.SOLVER.USE_SCHEDULER:
        from timm.scheduler import CosineLRScheduler
        schedulers = [CosineLRScheduler(optimizer,
                                        t_initial=cfg.SOLVER.ITER,
                                        lr_min=cfg.SOLVER.LR * cfg.SOLVER.LR_MIN_RATIO, 
                                        warmup_t=5,
                                        warmup_lr_init=cfg.SOLVER.LR * cfg.SOLVER.LR_MIN_RATIO,
                                        warmup_prefix=True)
                      for optimizer in model.optimizers]
    
    counter_early_stop = 0
    np.set_printoptions(precision=4)
    with tqdm(range(start_epoch, cfg.SOLVER.ITER)) as pbar:
        for i in pbar:
            if cfg.SOLVER.USE_SCHEDULER:
                [scheduler.step(i) for scheduler in schedulers]
                
            loss_list = []
            for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
                data_dict = dict_to_device(data_dict, cfg.DEVICE)
                loss_list.append(model.training_step(data_dict))
                
            loss_info = aggregate(loss_list)
            pbar.set_postfix(OrderedDict(loss_info))
            
            # validation
            if (i+1) % cfg.SOLVER.SAVE_EVERY == 0:
                if validation:
                    result_metrics = evaluate_model(cfg, model, val_data_loader)
                    print(result_metrics)
                    curr_val_score = result_metrics["score"]

                    if curr_val_score < val_score:
                        counter_early_stop = 0
                        val_score = curr_val_score
                        if save_model:
                            model.save(epoch=i)
                    else:
                        counter_early_stop += 1
                        if counter_early_stop == cfg.SOLVER.EARLY_STOP:
                            break
                            
                else:
                    if save_model:
                        model.save(epoch=i)
                
    if validation:
        return val_score

def evaluate_model(cfg: CfgNode, model: torch.nn.Module, data_loader):
    model.eval()
    metrics = build_metrics(cfg)
    
    result_metrics = {}

    with torch.no_grad():
        result_list = []
        for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
            data_dict = dict_to_device(data_dict, cfg.DEVICE)

            dict_list = []
            for _ in range(cfg.TEST.N_TRIAL):
                dict_list.append(copy_dict(model.predict(data_dict, return_prob=False)))
            
            dict_list = metrics.denormalize(dict_list) 
            result_list.append(copy_dict(metrics(dict_list)))
        
        d = aggregate(result_list)
        result_metrics.update({k: d[k] for k in d.keys() if d[k] != 0.0})
    
    model.train()

    return result_metrics

def dict_to_device(d, device):
    return {k: d[k].to(device) 
            if isinstance(d[k], torch.Tensor)
            else d[k]
            for k in d}


def copy_dict(d):
    return {k: d[k].clone()
            if isinstance(d[k], torch.Tensor)
            else d[k]
            for k in d}


def aggregate(dict_list: List[Dict]) -> Dict:
    ret_dict = {k: np.concatenate([d[k] if type(d[k]) == np.ndarray else [d[k]] for d in dict_list], axis=0) for k in dict_list[0].keys()}
    ret_dict = {k: np.nanmean(ret_dict[k]) for k in ret_dict}
    ret_dict = {k: float(ret_dict[k]) for k in ret_dict}
    for k in ret_dict:
        if k not in ['loss']:
            ret_dict[k] = np.round(ret_dict[k], 3)
    return ret_dict


if __name__ == "__main__":
    main()