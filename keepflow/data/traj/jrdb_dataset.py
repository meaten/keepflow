import sys
from pathlib import Path
from tqdm import tqdm
import itertools
from joblib import Parallel, delayed
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append('extern/traj')
from jrdb_traj.jrdb_baselines.trajnetbaselines.lstm.data_load_utils import prepare_data
from jrdb_traj.jrdb_baselines.trajnetbaselines.lstm.lstm import keep_valid_neigh, drop_distant_far
from jrdb_traj.jrdb_baselines.trajnetbaselines.lstm.tools.reader import Reader
# from jrdb_traj.jrdb_baselines.trajnetbaselines.lstm.utils import random_rotation


def jrdb_collate(batch):
    
    (index, obs, gt,
     neighbors, neighbors_gt,
     agent_type, neighbor_type,
     curr_state, robot_fut, map, dt, 
     first_history_index, this_node_type) = zip(*batch)
    
    obs = np.array(obs)
    gt = np.array(gt)
    
    num_neighbor_max = max([len(n) for n in neighbor_type])
    neighbors_ = np.zeros(obs.shape[:1] + (num_neighbor_max,) + obs.shape[1:]) * np.nan
    neighbors_gt_ = np.zeros(gt.shape[:1] + (num_neighbor_max,) + gt.shape[1:]) * np.nan
    for i in range(len(index)):
        neighbors_[i, :len(neighbors[i])] = neighbors[i]
        neighbors_gt_[i, :len(neighbors_gt[i])] = neighbors_gt[i]
        
    if len(neighbor_type) == 1:
        neighbor_type = np.array(neighbor_type)
    else:
        neighbor_type = np.column_stack(list(itertools.zip_longest(*neighbor_type, fillvalue=-1)))
    out = {
        "index": index,
        "obs": torch.Tensor(obs),
        "gt": torch.Tensor(gt),
        "neighbors": torch.Tensor(neighbors_), 
        "neighbors_gt": torch.Tensor(neighbors_gt_),
        'agent_type': agent_type,
        "neighbor_type": torch.Tensor(neighbor_type),
        "curr_state": torch.Tensor(np.array(curr_state)),
        "robot_traj": None,
        "map": None,
        "dt": torch.Tensor(np.array(dt)),
        "first_history_index": torch.LongTensor(np.array(first_history_index)),
        "this_node_type": torch.Tensor(this_node_type),
    }
    
    return out
    
    
class JRDB_Dataset(Dataset):
    def __init__(self, cfg, split):
        self.state = cfg.DATA.TRAJ.STATE 
        self.pred_state = cfg.DATA.TRAJ.PRED_STATE 
        
        path_npz = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'jrdb_cache'
        path_npz.mkdir(exist_ok=True)
        path_npz /= ('_'.join([cfg.DATA.DATASET_NAME, split]) + '.npz')
        
        if path_npz.exists():
            print(f'found serialized jrdb dataset at {path_npz}. loading...')
            with np.load(path_npz, allow_pickle=True, mmap_mode='r') as npz:
                self.idx = list(map(tuple, npz['idx']))
                self.obs = npz['obs']
                self.gt = npz['gt']
                self.neighbors = npz['neighbors']
                self.neighbors_gt = npz['neighbors_gt']
                self.agent_type = npz['agent_type']
                self.neighbor_type = npz['neighbor_type']
                self.curr_state = npz['curr_state']
                self.map = npz['map']
                self.robot_fut = npz['robot_fut']
                self.dt = npz['dt']
                self.first_history_index = npz['first_history_index']
                self.this_node_type = npz['this_node_type']
        else:
            print(f'creating serialized jrdb dataset at {path_npz}...')
            path = 'extern/traj/jrdb_traj/jrdb_baselines/DATA_BLOCK/jrdb_traj/'
            split = split + '/'
            dt = 0.5
            scenes, _, _ = prepare_data(path, subset=split, goals=False)
            # data = [paths_to_item(scene, dt) for scene in tqdm(scenes)]
            data = Parallel(n_jobs=4)(delayed(paths_to_item)(scene, dt) for scene in tqdm(scenes))
            data = [d for d in data if not len(d) == 0]
            idx, obs, gt, neighbors, neighbors_gt, \
                agent_type, neighbor_type, curr_state, \
                    dt, first_history_index = map(list, zip(*data))
            neighbors = np.array(neighbors, dtype=object)
            neighbors_gt = np.array(neighbors_gt, dtype=object)
            neighbor_type = np.array(neighbor_type, dtype=object)
            this_node_type = agent_type
        
            self.idx = idx
            self.obs = np.array(obs)
            self.gt = np.array(gt)
            self.neighbors = neighbors
            self.neighbors_gt = neighbors_gt
            self.agent_type = np.array(agent_type)
            self.neighbor_type = neighbor_type
            self.curr_state = np.array(curr_state)
            self.map = [None] * len(self.idx)
            self.robot_fut = [None] * len(self.idx)
            self.dt = np.array(dt)
            self.first_history_index = np.array(first_history_index)
            self.this_node_type = np.array(this_node_type)
            
            print('saving serialized jrdb as npz format...')
            np.savez(path_npz,
                     idx=self.idx,
                     obs=self.obs,
                     gt=gt,
                     neighbors=self.neighbors,
                     neighbors_gt=self.neighbors_gt,
                     agent_type=self.agent_type,
                     neighbor_type=self.neighbor_type,
                     curr_state=self.curr_state,
                     map=self.map,
                     robot_fut=self.robot_fut,
                     dt=self.dt,
                     first_history_index=self.first_history_index,
                     this_node_type=this_node_type)
            print('complete')
        
        jrdb_collate((self.__getitem__(0), self.__getitem__(1)))
        
    def __len__(self):
         return len(self.idx)

    def __getitem__(self, idx):
        obs = fetch(self.state, self.obs[idx])
        neighbors = fetch(self.state, self.neighbors[idx])
        gt = fetch(self.pred_state, self.gt[idx])
        neighbors_gt = fetch(self.pred_state, self.neighbors_gt[idx])
        return (self.idx[idx], obs, gt, \
            neighbors, neighbors_gt, \
            self.agent_type[idx], self.neighbor_type[idx],\
            self.curr_state[idx], self.map[idx], \
            self.robot_fut[idx], self.dt[idx], \
            self.first_history_index[idx], self.this_node_type[idx])
     
     
def paths_to_item(scene, dt):
    filename, scene_id, paths = scene
    ped_id = str(paths[0][0].pedestrian)
    paths = Reader.paths_to_xy(paths)
    paths = filter_path(paths)
    if len(paths) == 0:
        return []
    pos, vel, acc, nan_flag = calc_pos_vel_acc(paths, dt)
    
    curr_state = np.copy(pos[7, 0])
    pos -= curr_state
    pva = np.concatenate([pos, vel, acc], axis=-1)
    obs = pva[:8].transpose(1, 0, 2)
    gt = pva[8:-1].transpose(1, 0, 2)

    obs_ = obs[0]
    gt_ = gt[0]
    neighbors_ = obs[1:]
    neighbors_gt_ = gt[1:]
    curr_state = curr_state
    first_history_index = np.argmax(nan_flag[:, 0])

    idx = (filename, scene_id, ped_id)
    agent_type = 2
    neighbor_type = [2] * len(neighbors_)
    
    return idx, obs_, gt_, neighbors_, neighbors_gt_, agent_type, neighbor_type, curr_state, dt, first_history_index

def fetch(state, pva):
    if state == 'state_p':
        return pva[..., :2]
    elif state == 'state_v':
        return pva[..., 2:4]
    elif state == 'state_pva':
        return pva
    else:
        raise ValueError
     
def filter_path(paths):
    # assume obs length = 8
    if paths[9-2,0,-1]==0 or paths[9-1,0,-1]==0:
        return []
    paths = keep_valid_neigh(paths)
    paths, _ = drop_distant_far(paths, r=3.0)
    return paths


def calc_pos_vel_acc(paths, dt):
    pos = paths[..., :2]  # timesteps, n_agent, [x, y]
    nan_flag = paths[..., -1]
    nan_flag = np.nan_to_num(nan_flag, nan=0.0)
    
    vel = derivative_of(pos, nan_flag, dt)
    acc = derivative_of(vel, nan_flag, dt)

    return pos, vel, acc, nan_flag


def derivative_of(seq, nan_flag, dt):
    dseq = np.zeros_like(seq) * np.nan
    dseq[1:] = (seq[1:] - seq[:-1]) / dt
    
    idx_first_non_nan = np.argmax(nan_flag, axis=0)
    dseq[idx_first_non_nan] = dseq[idx_first_non_nan + 1]
    
    return dseq