# I created this data loader by refering following great reseaches & github repos.
# video: stochastic video generation https://github.com/edenton/svg
# motion: On human motion prediction using recurrent neural network https://github.com/wei-mao-2019/LearnTrajDep
# traj: Social GAN https://github.com/agrimgupta92/sgan
#     Trajectron++ https://github.com/StanfordASL/Trajectron-plus-plus
#     Motion Indeterminacy Diffusion https://github.com/gutianpei/mid

from typing import Tuple
from collections.abc import Callable
from pathlib import Path
import dill
from yacs.config import CfgNode
from torch.utils.data import Dataset, DataLoader
from trajdata import UnifiedDataset


def build_dataloader(cfg: CfgNode, rand=True, split="train", batch_size=None) -> DataLoader:
    # train, val, test
    if cfg.DATA.TASK == "traj":
        dataset, collate_fn = build_traj_dataloader(cfg, split)
    elif cfg.DATA.TASK == "motion":
        dataset, collate_fn = build_motion_dataloader(cfg, split)
    elif cfg.DATA.TASK == "video":
        dataset, collate_fn = build_video_dataloader(cfg, split)
    else:
        raise ValueError        
    
    if batch_size is None:
        batch_size = cfg.DATA.BATCH_SIZE
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=rand,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True if split == 'train' else False,
        pin_memory=True)
    
    return loader


def build_traj_dataloader(cfg: CfgNode, split: str) -> Tuple[Dataset, Callable] :
    # train, val, test
    if 'sim' in cfg.DATA.DATASET_NAME:
        from .traj.trajectron_dataset import EnvironmentDataset, get_hypers
        hypers = get_hypers(cfg)        
        
        env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'processed_data' / f"{cfg.DATA.DATASET_NAME}_{split}.pkl"
        with open(env_path, 'rb') as f:
            env = dill.load(f, encoding='latin1')

        dataset = EnvironmentDataset(env,
                                    state=hypers[cfg.DATA.TRAJ.STATE],
                                    pred_state=hypers[cfg.DATA.TRAJ.PRED_STATE],
                                    node_freq_mult=hypers['scene_freq_mult_train'],
                                    scene_freq_mult=hypers['node_freq_mult_train'],
                                    hyperparams=hypers,
                                    min_history_timesteps=1 if cfg.DATA.TRAJ.ACCEPT_NAN and split == 'train' else cfg.DATA.OBSERVE_LENGTH,
                                    min_future_timesteps=cfg.DATA.PREDICT_LENGTH,
                                    #augment=hypers['augment'] and split == 'train'
                                    )
        
        for d in dataset:
            if d.node_type == 'PEDESTRIAN':
                dataset = d
                break

        from .traj.preprocessing import dict_collate as seq_collate
            
    elif cfg.DATA.DATASET_NAME == 'jrdb':
        from .traj.jrdb_dataset import JRDB_Dataset
        dataset = JRDB_Dataset(cfg, split)
        from .traj.jrdb_dataset import jrdb_collate as seq_collate
    else:
        agent_interaction_distances = {(2,2): 3.0}
        
        if cfg.DATA.DATASET_NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
            desired_dt = 0.4
            history_sec =(desired_dt, (cfg.DATA.OBSERVE_LENGTH - 1) * desired_dt) if cfg.DATA.TRAJ.ACCEPT_NAN and split == 'train' else ((cfg.DATA.OBSERVE_LENGTH - 1) * desired_dt,) * 2
            future_sec = ((cfg.DATA.PREDICT_LENGTH) * desired_dt,) * 2
            agent_interaction_distances = {(2,2): 3.0}
            # state_format = "x,y,xd,yd,xdd,ydd"
            # obs_format = "x,y,xd,yd,xdd,ydd"
            
            desired_data = f"eupeds_{cfg.DATA.DATASET_NAME}"
            desired_split = desired_data + f"-{split}_loo"
            data_dir = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'raw' / 'all_data'
            
        dataset = UnifiedDataset(
            desired_data=[desired_split],
            data_dirs={  # Remember to change this to match your filesystem!
                desired_data: data_dir
            },
            history_sec=history_sec,
            future_sec=future_sec,
            agent_interaction_distances = agent_interaction_distances,
            # state_format=state_format,  # cause error
            # obs_format=obs_format,
            desired_dt=desired_dt,
            standardize_data=False,
            centric='agent'
        )
        cache_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'cache'
        cache_path.mkdir(exist_ok=True)
        cache_path = cache_path / desired_split 
        dataset.load_or_create_cache(
            str(cache_path), num_workers=cfg.DATA.NUM_WORKERS, filter_fn=None
        )

        from .traj.collate_wrapper import CollateWrapper
        if cfg.DATA.DATASET_NAME == 'eth' and split == 'test':
            seq_collate = CollateWrapper(cfg, dataset.get_collate_fn(), scale=0.6)
        else:
            seq_collate = CollateWrapper(cfg, dataset.get_collate_fn(), scale=1.0)

    return dataset, seq_collate


def build_motion_dataloader(cfg: CfgNode, split: str) -> Tuple[Dataset, Callable]:
    # train, val, test
    if cfg.DATA.DATASET_NAME == "h36motion":
        import os
        from data.motion.h36motion import H36motion
        dataset_train = H36motion(
            path_to_data=os.path.join(cfg.DATA.PATH, cfg.DATA.TASK, "h3.6m", "dataset"),
            actions="all",
            input_n=cfg.DATA.OBSERVE_LENGTH,
            output_n=cfg.DATA.PREDICT_LENGTH,
            split=0,
            load_3d=False)

        if split == "train":
            dataset = dataset_train
        elif split == "val":
            dataset_val = H36motion(
                path_to_data=os.path.join(cfg.DATA.PATH, cfg.DATA.TASK, "h3.6m", "dataset"),
                actions="smoking",
                input_n=cfg.DATA.OBSERVE_LENGTH,
                output_n=cfg.DATA.PREDICT_LENGTH,
                split=2,
                data_mean=dataset_train.data_mean,
                data_std=dataset_train.data_std,
                onehotencoder=dataset_train.onehotencoder,
                load_3d=False)

            dataset = dataset_val
        elif split == "test":
            dataset_test = H36motion(
                path_to_data=os.path.join(cfg.DATA.PATH, cfg.DATA.TASK, "h3.6m", "dataset"),
                actions="smoking",
                input_n=cfg.DATA.OBSERVE_LENGTH,
                output_n=cfg.DATA.PREDICT_LENGTH,
                split=1,
                data_mean=dataset_train.data_mean,
                data_std=dataset_train.data_std,
                onehotencoder=dataset_train.onehotencoder,
                load_3d=False)

            dataset = dataset_test
        
    from .motion.h36motion import seq_collate
        
    return dataset, seq_collate


def build_video_dataloader(cfg: CfgNode, split: str) -> Tuple[Dataset, Callable]:
    # only train, test
    from data.video.datasets_factory import video_dataset
    if cfg.DATA.DATASET_NAME == "bair":
        path = Path(cfg.DATA.PATH) / cfg.DATA.TASK
        img_width = 64
    elif cfg.DATA.DATASET_NAME == "kth":
        path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "kth_action"
        img_width = 128
    elif cfg.DATA.DATASET_NAME == "mnist":
        path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "moving-mnist-example" / f"moving-mnist-{split}.npz"
        img_width = 64
    dataset = video_dataset(dataset_name=cfg.DATA.DATASET_NAME,
                        data_path=path,
                        split=split,
                        img_width=img_width,
                        input_n=cfg.DATA.OBSERVE_LENGTH,
                        output_n=cfg.DATA.PREDICT_LENGTH,
                        injection_action="concat")
    
    from .video.mnist import seq_collate
    
    return dataset, seq_collate
        