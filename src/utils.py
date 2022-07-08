import os
import argparse
from yacs.config import CfgNode
import shutil
import torch

def load_config(args: argparse.Namespace) -> CfgNode:
    from default_params import _C as cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    cfg_ = cfg.clone()
    if os.path.isfile(args.config_file):
        conf = args.config_file
        print(f"Configuration file loaded from {conf}.")
        cfg_.merge_from_file(conf)
        cfg_.OUTPUT_DIR = os.path.join(cfg_.OUTPUT_DIR, 
                                       cfg_.DATA.TASK, 
                                       cfg_.DATA.DATASET_NAME,
                                       cfg_.MODEL.TYPE)
    else:
        print("Use default configuration.")
        cfg_.OUTPUT_DIR = os.path.join(cfg_.OUTPUT_DIR, "default")
    
    if cfg_.LOAD_TUNED and args.mode != "tune":
        cfg_ = load_tuned(args, cfg_)
    cfg_.freeze()
    
    print(f"output dirname: {cfg_.OUTPUT_DIR}")
    os.makedirs(cfg_.OUTPUT_DIR, exist_ok=True)
    if os.path.isfile(args.config_file):
        shutil.copy2(args.config_file, os.path.join(cfg_.OUTPUT_DIR, 'config.yaml'))

    return cfg_


def load_tuned(args: argparse.Namespace, cfg: CfgNode) -> CfgNode:
    print("load params from optuna database")
    import optuna
    study = optuna.load_study(storage=os.path.join("sqlite:///", cfg.OUTPUT_DIR, "optuna.db"), study_name="my_opt")

    trial_dict = study.best_trial.params
    
    for key in list(trial_dict.keys()):
        if type(trial_dict[key]) == str:
            exec(f"cfg.{key} = '{trial_dict[key]}'")
        else:
            exec(f"cfg.{key} = {trial_dict[key]}")
    
    return cfg


def optimizer_to_cuda(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()