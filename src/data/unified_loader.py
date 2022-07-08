# I created this data loader by refering following great reseaches & github repos.
# VP: stochastic video generation https://github.com/edenton/svg
# MP: On human motion prediction using recurrent neural network 
# https://github.com/wei-mao-2019/LearnTrajDep
# TP: Social GAN https://github.com/agrimgupta92/sgan
from pathlib import Path
from yacs.config import CfgNode
import torch
from torch.utils.data import DataLoader


def unified_loader(cfg: CfgNode, rand=True, split="train") -> DataLoader:
    # train, val, test
    if cfg.DATA.TASK == "TP":
        if cfg.DATA.DATASET_NAME in ["eth", "hotel", "univ", "zara1", "zara2"]:
            from data.TP.trajectories import TrajectoryDataset as TP_dataset
            path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / cfg.DATA.DATASET_NAME / split
            dataset = TP_dataset(
                path,
                obs_len=cfg.DATA.OBSERVE_LENGTH,
                pred_len=cfg.DATA.PREDICT_LENGTH,
                skip=cfg.DATA.SKIP,
                delim="\t"
            )
        elif cfg.DATA.DATASET_NAME == "simfork":
            from data.TP.trajectories import SimulatedForkTrajectory
            dataset = SimulatedForkTrajectory(
                num_data=1000 if split == "train" else 10,
                obs_len=cfg.DATA.OBSERVE_LENGTH,
                pred_len=cfg.DATA.PREDICT_LENGTH
            )
            
    # only train, test
    elif cfg.DATA.TASK == "VP":
        from data.VP.datasets_factory import VP_dataset
        if cfg.DATA.DATASET_NAME == "bair":
            path = Path(cfg.DATA.PATH) / cfg.DATA.TASK
            img_width = 64
        elif cfg.DATA.DATASET_NAME == "kth":
            path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "kth_action"
            img_width = 128
        elif cfg.DATA.DATASET_NAME == "mnist":
            path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "moving-mnist-example" / f"moving-mnist-{split}.npz"
            img_width = 64
        dataset = VP_dataset(dataset_name=cfg.DATA.DATASET_NAME,
                            data_path=path,
                            split=split,
                            img_width=img_width,
                            input_n=cfg.DATA.OBSERVE_LENGTH,
                            output_n=cfg.DATA.PREDICT_LENGTH,
                            injection_action="concat")
                                
                            

    # train, val, test
    elif cfg.DATA.TASK == "MP":
        if cfg.DATA.DATASET_NAME == "h36motion":
            import os
            from data.MP.h36motion import H36motion
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
            


    if cfg.DATA.TASK == "TP":
        from data.TP.trajectories import seq_collate
    elif cfg.DATA.TASK == "VP":
        from data.VP.mnist import seq_collate
    elif cfg.DATA.TASK == "MP":
        from data.MP.h36motion import seq_collate
    collate_fn = seq_collate
    

    batch_size = cfg.DATA.BATCH_SIZE if not split == "test" else cfg.DATA.BATCH_SIZE_TEST
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=rand,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True)

    return loader


