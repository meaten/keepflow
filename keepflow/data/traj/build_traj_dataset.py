from typing import Tuple
from collections.abc import Callable
from itertools import product
from pathlib import Path
import dill
from yacs.config import CfgNode
from torch.utils.data import Dataset

from trajdata import UnifiedDataset
from trajdata.data_structures import AgentType
from trajdata.augmentation import NoiseHistories


def build_traj_dataset(cfg: CfgNode, split: str) -> Tuple[Dataset, Callable] :
    # train, val, test
    if 'sim' in cfg.DATA.DATASET_NAME:
        from .trajectron_dataset import EnvironmentDataset, get_hypers
        hypers = get_hypers(cfg)
        
        if cfg.DATA.DATASET_NAME == 'sdd' and split == 'val':
            split = 'test'
        env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'processed_data' / f"{cfg.DATA.DATASET_NAME}_{split}.pkl"
        with open(env_path, 'rb') as f:
            env = dill.load(f, encoding='latin1')
                
        dataset = EnvironmentDataset(env,
                                    state={type: hypers[cfg.DATA.TRAJ.STATE] for type in env.NodeType},
                                    pred_state={type: hypers[cfg.DATA.TRAJ.PRED_STATE] for type in env.NodeType},
                                    node_freq_mult=hypers['scene_freq_mult_train'],
                                    scene_freq_mult=hypers['node_freq_mult_train'],
                                    hyperparams=hypers,
                                    # min_history_timesteps=3 if cfg.DATA.TRAJ.ACCEPT_NAN and split == 'train' else cfg.DATA.OBSERVE_LENGTH - 1,
                                    min_history_timesteps=cfg.DATA.OBSERVE_LENGTH-1,
                                    min_future_timesteps=cfg.DATA.PREDICT_LENGTH,
                                    #augment=hypers['augment'] and split == 'train'
                                    )
        
        for d in dataset:
            if d.node_type == 'PEDESTRIAN':
                dataset = d
                break
        
        from .preprocessing import dict_collate as seq_collate
            
    elif cfg.DATA.DATASET_NAME == 'jrdb':    
        from .jrdb_dataset import JRDB_Dataset
        dataset = JRDB_Dataset(cfg, split)
        from .jrdb_dataset import jrdb_collate as seq_collate
    else:
        params = load_params(cfg)
        desired_data, desired_split, data_dir = load_path(cfg, split)
            
        cache_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'cache'
            
        noise_hists = NoiseHistories()
        
        # Standardize all data such that
        # (1) the predicted agent's orientation at the current timestep is 0
        # (2) all data is made relative to the predicted agent's current position 
        # and (3) the agent's heading value is replaced with its sin, cos values. Defaults to True.
        standardize_data = True

        # Make agent velocities and accelerations relative to the agent being predicted. Defaults to False.
        standardize_derivatives = False

        sec_two_steps = round(2 * params['desired_dt'], 1)
        
        dataset = UnifiedDataset(
            desired_data=[desired_split],
            data_dirs={  # Remember to change this to match your filesystem!
                desired_data: data_dir
            },

            # history_sec=params['history_sec'],
            history_sec=(sec_two_steps, params['history_sec'][0]) if cfg.DATA.TRAJ.ACCEPT_NAN and split == 'train' else params['history_sec'],
            future_sec=params['future_sec'],
            agent_interaction_distances=params["agent_interaction_distances"],
            only_types=params["agent_list"],
            only_predict=params["only_predict_list"],
            desired_dt=params["desired_dt"],
            standardize_data=standardize_data,
            standardize_derivatives=standardize_derivatives,
            centric='agent',
            augmentations=[noise_hists] if split == 'train' and cfg.DATA.AUGMENTATION else None,
            cache_location=str(cache_path),
            incl_raster_map=cfg.DATA.TRAJ.USE_MAP,
            raster_map_params=params["raster_map_params"],
            # rebuild_cache=True,
            # rebuild_maps=True,
            # verbose=True
        )
        
        from .collate_wrapper import CollateWrapper
        if cfg.DATA.DATASET_NAME == 'eth' and split == 'test':
            scale = cfg.DATA.TRAJ.SCALE * 0.6  #HACK
            # scale = cfg.DATA.TRAJ.SCALE
        else:
            scale = cfg.DATA.TRAJ.SCALE
        seq_collate = CollateWrapper(cfg, dataset.get_collate_fn(),
                                     scale=scale,
                                     standardize_data=standardize_data,
                                     standardize_derivatives=standardize_derivatives)
    
    return dataset, seq_collate


def load_path(cfg: CfgNode, split: str) -> Tuple[str, Path]:
    if cfg.DATA.DATASET_NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        desired_data = f"eupeds_{cfg.DATA.DATASET_NAME}"
        desired_split = desired_data + f"-{split}_loo"
        data_dir = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'eth_ucy'
    
    # turned out to be different settings from the conventional evaluation
    elif cfg.DATA.DATASET_NAME == 'sdd':
        if split == 'val':
            split = 'test'
        desired_data = "sdd_trajnet"
        desired_split = desired_data + f"-{split}"
        data_dir = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'StanfordDroneDataset'
        
    elif cfg.DATA.DATASET_NAME == 'nuscenes':
        if split == 'test':
            split = 'val'
        desired_data = f"nusc_trainval"
        desired_split = desired_data + f"-{split}"
        data_dir = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'nuScenes'
        
    return desired_data, desired_split, data_dir


def load_params(cfg: CfgNode) -> dict:
    if "sim" in cfg.DATA.DATASET_NAME:
        params = params_sim()
    elif cfg.DATA.DATASET_NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        params = params_ethucy()
    elif cfg.DATA.DATASET_NAME == 'sdd':
        params = params_sdd()
    elif cfg.DATA.DATASET_NAME == 'nuscenes':
        params = params_nuscenes()

    return params

def params_sim():
    return {"desired_dt": 0.4,
            "history_sec": (2.8, 2.8),
            "future_sec": (4.8, 4.8),
            "agent_list": [AgentType.PEDESTRIAN],
            "agent_interaction_distances": {(2,2): 3.0},
            "only_predict_list": [AgentType.PEDESTRIAN],
            "raster_map_params": None}

def params_ethucy():
    return {"desired_dt": 0.4,
            "history_sec": (2.8, 2.8),
            "future_sec": (4.8, 4.8),
            "agent_list": [AgentType.PEDESTRIAN],
            "agent_interaction_distances": {(2,2): 3.0},
            "only_predict_list": [AgentType.PEDESTRIAN],
            "raster_map_params": None}
    
def params_sdd():
    agent_list = [AgentType.UNKNOWN, AgentType.PEDESTRIAN, AgentType.BICYCLE, AgentType.VEHICLE]
    return {"desired_dt": 0.4,
            "history_sec": (2.8, 2.8),
            "future_sec": (4.8, 4.8),
            "agent_list": agent_list,
            "agent_interaction_distances": {c: 10.0 for c in product(agent_list, repeat=2)},
            "only_predict_list": [AgentType.PEDESTRIAN],
            "raster_map_params": {'px_per_m': 20, 'map_size_px': 100, "offset_frac_xy": (-0.5, 0.0),},
            }
    
def params_nuscenes():
    agent_list = [AgentType.UNKNOWN, AgentType.PEDESTRIAN, AgentType.BICYCLE, AgentType.VEHICLE]
    return {"desired_dt": 0.5,
            "history_sec": (2.0, 2.0),
            "future_sec": (3.0, 3.0),
            "agent_list": agent_list,
            "agent_interaction_distances": {c: 20.0 for c in product(agent_list, repeat=2)},
            "only_predict_list": [AgentType.VEHICLE],
            "raster_map_params": None,
            }
