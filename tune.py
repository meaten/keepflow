import os
import argparse
from yacs.config import CfgNode

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import optuna

from keepflow.utils import load_config
from train import train


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    tune(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch hyperparameter tuning code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--device", type=str, default="cuda:0")
    
    return parser.parse_args()
    
    
def tune(cfg: CfgNode) -> None:
    
    def objective_with_arg(cfg):
        _cfg = cfg.clone()
        _cfg.defrost()
        
        def objective(trial):
            _cfg.MODEL.FLOW.N_HIDDEN = trial.suggest_int(
                "MODEL.FLOW.N_HIDDEN", 1, 3)
            _cfg.MODEL.FLOW.HIDDEN_SIZE = trial.suggest_int(
                "MODEL.FLOW.HIDDEN_SIZE", 16, 128, step=16)
            _cfg.MODEL.FLOW.COND_SIZE = trial.suggest_int(
                "MODEL.FLOW.COND_SIZE", 16, 128, step=16)
            _cfg.MODEL.FLOW.BASE_SIGMA = trial.suggest_float(
                "MODEL.FLOW.BASE_SIGMA", 1e-3, 1e-0, log=True)
            _cfg.SOLVER.LR = trial.suggest_float(
                "SOLVER.LR", 1e-5, 1e-3, log=True)
            
            return train(_cfg, save_model=False, load_model=False)
    
        return objective

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    
    study = optuna.create_study(sampler=sampler, pruner=pruner,
                               direction='minimize',
                               storage=os.path.join(
                                   "sqlite:///", cfg.SAVE_DIR, "optuna.db"),
                               study_name='my_opt',
                               load_if_exists=True)
    study.optimize(objective_with_arg(cfg), n_jobs=1, n_trials=200, gc_after_trial=True)
    
    trial = study.best_trial

    print(trial.value, trial.params)
    

if __name__ == "__main__":
    main()