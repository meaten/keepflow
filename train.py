import argparse
from typing import List, Dict
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
from yacs.config import CfgNode
import numpy as np
import torch

from keepflow.utils import load_config
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch training code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--device", type=str, default="cuda:0")
    
    return parser.parse_args()


def train(cfg: CfgNode, save_model=True) -> None:        
    validation = cfg.SOLVER.VALIDATION and cfg.DATA.TASK != "video" 
    
    data_loader = build_dataloader(cfg, rand=True, split="train")
    # we don't have any validation set for Video Prediction
    if validation:
        val_data_loader = build_dataloader(cfg, rand=False, split="val")
        val_score = np.inf
    
    start_epoch = 0
    model = build_model(cfg)
    
    if model.check_saved_path():
        # model saved at the end of each epoch.
        # If model parameters are saved, load and resume training from next epoch
        start_epoch = model.load() + 1
        print('loaded pretrained model')

    if cfg.SOLVER.USE_SCHEDULER:
        schedulers = [torch.optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=int(cfg.SOLVER.ITER/10),
                                                      last_epoch=start_epoch-1,
                                                      gamma=0.7) for optimizer in model.optimizers]
        
    np.set_printoptions(precision=4)
    with tqdm(range(start_epoch, cfg.SOLVER.ITER)) as pbar:
        for i in pbar:
            loss_list = []
            for data_dict in tqdm(data_loader, leave=False):
                data_dict = {k: data_dict[k].to(cfg.DEVICE) 
                            if isinstance(data_dict[k], torch.Tensor)
                            else data_dict[k]
                            for k in data_dict}
                
                loss_list.append(model.update(deepcopy(data_dict)))
                
            loss_info = aggregate(loss_list)
            pbar.set_postfix(OrderedDict(loss_info))

            # validation
            if (i+1) % cfg.SOLVER.SAVE_EVERY == 0:
                if validation:
                    result_metrics = evaluate_model(cfg, model, val_data_loader)
                    print(result_metrics)
                    curr_val_score = result_metrics["score"]

                    if curr_val_score < val_score:
                        val_score = curr_val_score
                        if save_model:
                            model.save(epoch=i)
                else:
                    if save_model:
                        model.save(epoch=i)
            
            if cfg.SOLVER.USE_SCHEDULER:
                [scheduler.step() for scheduler in schedulers]
                
    if validation:
        return val_score

def evaluate_model(cfg: CfgNode, model: torch.nn.Module, data_loader):
    model.eval()
    metrics = build_metrics(cfg)
    
    result_metrics = {}

    with torch.no_grad():
        result_list = []
        for i, data_dict in enumerate(tqdm(data_loader, leave=False)):
            data_dict = {k: data_dict[k].to(cfg.DEVICE) 
                        if isinstance(data_dict[k], torch.Tensor)
                        else data_dict[k]
                        for k in data_dict}
            
            dict_list = []
            for _ in range(cfg.TEST.N_TRIAL):
                result_dict = model.predict(data_dict, return_prob=False)
                dict_list.append(deepcopy(result_dict))
            
            dict_list = metrics.denormalize(dict_list) 
            result_list.append(deepcopy(metrics(dict_list)))
        d = aggregate(result_list)
        result_metrics.update({k: d[k] for k in d.keys() if d[k] != 0.0})
    
    model.train()

    return result_metrics
        
        
def aggregate(dict_list: List[Dict]) -> Dict:
    ret_dict = {k: np.concatenate([d[k] if type(d[k]) == np.ndarray else [d[k]] for d in dict_list], axis=0) for k in dict_list[0].keys()}
    ret_dict = {k: np.nanmean(ret_dict[k]) for k in ret_dict}
    ret_dict = {k: float(ret_dict[k]) for k in ret_dict}
    return ret_dict


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    train(cfg)

if __name__ == "__main__":
    main()