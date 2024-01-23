import os
import argparse
from yacs.config import CfgNode
import shutil
import optuna


def load_config(args: argparse.Namespace, while_tuning: bool=False) -> CfgNode:
    from .default_params import _C as cfg
    
    cfg_ = cfg.clone()
    if os.path.isfile(args.config_file):
        conf = args.config_file
        print(f"Configuration file loaded from {conf}.")
        cfg_.merge_from_file(conf)
        cfg_.SAVE_DIR = os.path.join(cfg_.SAVE_DIR, 
                                       os.path.splitext(conf)[0])
                                       
    else:
        raise FileNotFoundError
    
    if cfg_.LOAD_TUNED and not while_tuning:
        cfg_ = load_tuned_params(args, cfg_)
        
    # only to(device) cannot force trajdata on the specified device
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cfg_.DEVICE = args.device
    if "cuda" in args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
        cfg_.DEVICE = "cuda:0"
    cfg_.freeze()
    
    print(f"output dirname: {cfg_.SAVE_DIR}")
    os.makedirs(cfg_.SAVE_DIR, exist_ok=True)
    if os.path.isfile(args.config_file):
        shutil.copy2(args.config_file, os.path.join(cfg_.SAVE_DIR, 'config.yaml'))

    return cfg_


def load_tuned_params(args: argparse.Namespace, cfg: CfgNode) -> CfgNode:
    study_path = os.path.join(cfg.SAVE_DIR, "optuna.db")
    if not os.path.exists(study_path):
        return cfg
    
    study_path = os.path.join("sqlite:///", study_path)
    print("load hyperparameters from optuna database")
    study = optuna.load_study(storage=study_path, study_name="my_opt")
    trial_dict = study.best_trial.params
    
    for key in list(trial_dict.keys()):
        if type(trial_dict[key]) == str:
            exec(f"cfg.{key} = '{trial_dict[key]}'")
        else:
            exec(f"cfg.{key} = {trial_dict[key]}")
    
    return cfg