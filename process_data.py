import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
np.random.seed(123)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import dill
import torch
import torch.distributions as D

from keepflow.data.traj.environment import Environment, Scene, Node, derivative_of
from keepflow.visualization import plot_density
from keepflow.visualization.traj.traj_visualizer import get_grid
from keepflow.utils.kde import GaussianKDE

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.4

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 1}, # 2 -> 1
            'y': {'mean': 0, 'std': 1} 
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

nl = 0
l = 0

# raw_path = './keepflow/data/traj/raw_data/'
# data_folder_name = './keepflow/data/traj/processed_data/'

raw_path = Path('/home-local/maeda/keepflow/data/traj/')
output_folder_name = Path('/home-local/maeda/keepflow/data/traj/processed_data/')

os.makedirs(output_folder_name, exist_ok=True)

data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


def preprocess_node(data, node_id, dt):
    node_df = data[data['node_id'] == node_id]

    node_values = node_df[['pos_x', 'pos_y']].values

    new_first_idx = node_df['frame_id'].iloc[0]

    x = node_values[:, 0]
    y = node_values[:, 1]
    
    vx = derivative_of(x, dt)
    vy = derivative_of(y, dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)

    data_dict = {('position', 'x'): x,
                ('position', 'y'): y,
                ('velocity', 'x'): vx,
                ('velocity', 'y'): vy,
                ('acceleration', 'x'): ax,
                ('acceleration', 'y'): ay}

    node_data = pd.DataFrame(data_dict, columns=data_columns)
    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
    node.first_timestep = new_first_idx

    return node

def plot_traj(obs, gt, path: Path, max_pos, min_pos) -> None:
    f, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_pos[0], max_pos[0])
    ax.set_ylim(min_pos[1], max_pos[1])
    
    sns.lineplot(x=obs[:, 0], y=obs[:, 1], color='black',
                    legend='brief', label="obs", marker='o', sort=False)
    
    shape = np.array(gt.shape)
    shape[1] += 1
    gt_vis = np.zeros(shape)
    gt_vis[:, 0] = obs[-1]
    gt_vis[:, 1:] = gt
    for i in range(gt_vis.shape[0]):
        if i == 0:
            sns.lineplot(x=gt_vis[i, :, 0], y=gt_vis[i, :, 1],
                            color='blue', legend='brief', label="GT", marker='o', sort=False)
        else:
            sns.lineplot(
                x=gt_vis[i, :, 0], y=gt_vis[i, :, 1], color='blue', marker='o', sort=False)

    plt.legend()
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_dot(point: np.array, path: Path, max_pos, min_pos) -> None:    
    f, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_pos[0], max_pos[0])
    ax.set_ylim(min_pos[1], max_pos[1])
    
    plt.plot(point[0], point[1], 'o', markersize=6, color='black')
    
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def visualize_env(env, path):
    obs_len = 8
    pred_len = 12
    
    os.makedirs(path, exist_ok=True)
    # traj
    scene = env.scenes[0]
    max_pos, min_pos = scene.calculate_pos_min_max()
    max_pos += 0.05 * (max_pos - min_pos)
    min_pos -= 0.05 * (max_pos - min_pos)
    obs = []
    gt = []
    for i in np.random.permutation(len(scene.nodes))[:50]:
        obs.append(scene.nodes[i].data.data[:obs_len, :2])
        gt.append(scene.nodes[i].data.data[obs_len:, :2])
    obs = np.array(obs)
    gt = np.array(gt)
    plot_traj(obs[0], gt, path / 'traj.png', max_pos, min_pos)
    
    # density
    path_density = path / 'density'
    os.makedirs(path_density, exist_ok=True)
    xx, yy = get_grid(500, min_pos, max_pos)
    if env.gt_dist_type == 'gmm':
        value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[None, :, None].tile(1, 1, pred_len, 1)
        log_prob = env.gt_dist.log_prob(value)
        prob = torch.exp(log_prob)
    elif env.gt_dist_type == 'kde':
        value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1).cuda()
        prob = []
        for i in range(pred_len):
            prob.append(env.gt_dist[i](value))
        prob = torch.stack(prob, dim=1)[None].cpu()
    else:
        raise NotImplementedError
    for i in range(obs_len):
        plot_dot(obs[0][i], path_density / f'{i:02}.png', max_pos, min_pos)
    for i in range(pred_len):
        prob[:, :, i] /= prob[:, :, i].max()
        plot_density(xx, yy, prob[:, :, i].detach().numpy(), path_density / f'{i + obs_len:02}.png')
    plot_density(xx, yy, prob.mean(dim=2).detach().numpy(), path_density / 'mean.png')
    
    # updated density
    path_updated_density = path / 'updated_density'
    os.makedirs(path_updated_density, exist_ok=True)
    update_step = 2
    if env.gt_dist_type == 'gmm':
        means = env.gt_dist.component_distribution.mean
        stds = torch.sqrt(env.gt_dist.component_distribution.variance)
        _, N_modes, _ = means.shape
        costs = (means[:update_step] - gt[0, :update_step, None]).norm(dim=-1).sum(dim=0)
        idxs = torch.argsort(costs)[:N_modes // 2]
        mix = D.Categorical(torch.ones(N_modes // 2))
        comp = D.Independent(D.Normal(means[update_step:, idxs], stds[update_step:, idxs]), 1)
        gt_dist_updated = D.MixtureSameFamily(mix, comp)
        log_prob = gt_dist_updated.log_prob(value[:, :, update_step:])
        prob = torch.exp(log_prob)
    elif env.gt_dist_type == 'kde':
        num_samples = 300
        traj = torch.stack([torch.matmul(env.gt_dist[i].dataset, env.gt_dist[i].bw_transform.inverse())
                            for i in range(update_step)], dim=1)
        costs = (traj.cpu() - gt[[0], :update_step]).norm(dim=-1).sum(dim=1)
        idxs = torch.argsort(costs)[:num_samples]
        gt_dist_updated = [GaussianKDE(torch.matmul(env.gt_dist[i].dataset[idxs], env.gt_dist[i].bw_transform.inverse()))
                           for i in range(update_step, pred_len)]
        prob = torch.stack([gt_dist_updated[i](value) for i in range(pred_len - update_step)], dim=1)[None].cpu()
    else:
        raise NotImplementedError
    for i in range(pred_len - update_step): 
        prob[:, :, i] /= prob[:, :, i].max()   
        plot_density(xx, yy, prob[:, :, i].detach().numpy(), path_updated_density / f'{i+obs_len+update_step:02}.png')
    plot_density(xx, yy, prob.mean(dim=2).detach().numpy(), path_updated_density / 'mean.png')
    

# Process simulated dataset
for desired_source in ['simfork', 'simbranch']:
# for desired_source in [ 'simfork']:
    for data_class in ['train', 'val', 'test']:
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = output_folder_name / ('_'.join([desired_source, data_class]) + '.pkl')

        d = {'train': 10000,
             'val': 1000,
             'test': 1000}
        N_seqs = d[data_class]
        obs_len = 8
        pred_len = 12
        seq_len = obs_len + pred_len
        theta = np.pi / 6
        std = 0.3
        
        traj_base = []
        for sgn in [1, -1]:
            traj_flat_x = np.arange(-obs_len+1, 1., 1., dtype=float)
            traj_flat_y = np.zeros_like(traj_flat_x, dtype=float)
            traj_flat = np.concatenate([traj_flat_x[:, None], traj_flat_y[:, None]], axis=1)[None]
            
            if desired_source == 'simline':
                dist = np.arange(1., pred_len+1., 1., dtype=float)
                traj_fut_x = dist
                traj_fut_y = np.zeros_like(traj_fut_x)
                traj_fut = np.concatenate([traj_fut_x[:, None], traj_fut_y[:, None]], axis=1)[None]
            elif desired_source == 'simfork':
                dist = np.arange(1., pred_len+1., 1., dtype=float)
                traj_fut_x = dist * np.cos(sgn * theta, dtype=float)
                traj_fut_y = dist * np.sin(sgn * theta, dtype=float)
                traj_fut = np.concatenate([traj_fut_x[:, None], traj_fut_y[:, None]], axis=1)[None]
            elif desired_source == 'simcross':
                dist = np.arange(1., pred_len+1., 1., dtype=float)
                traj_fut_x = dist * np.cos(sgn * theta, dtype=float)
                dist = np.concatenate([np.arange(1., pred_len // 3 + 1., 1., dtype=float),
                                       np.arange(pred_len // 3 - 1., -pred_len // 3 + 1., -1., dtype=float)])
                traj_fut_y = dist * np.sin(sgn * theta, dtype=float)
                traj_fut = np.concatenate([traj_fut_x[:, None], traj_fut_y[:, None]], axis=1)[None]
            elif desired_source == 'simbranch':
                dist = np.arange(1., pred_len+1., 1., dtype=float)
                traj_fut_x = dist * np.cos(sgn * theta, dtype=float)
                traj_fut_y1 = dist * np.sin(sgn * theta, dtype=float)
                dist = np.concatenate([np.arange(1., pred_len // 2 + 1, 1., dtype=float),
                                       np.arange(pred_len // 2 + 1, 1., -1., dtype=float)])
                traj_fut_y2 = dist * np.sin(sgn * theta, dtype=float)
                traj_fut1 = np.concatenate([traj_fut_x[:, None], traj_fut_y1[:, None]], axis=1)[None]
                traj_fut2 = np.concatenate([traj_fut_x[:, None], traj_fut_y2[:, None]], axis=1)[None]
                traj_fut = np.concatenate([traj_fut1, traj_fut2], axis=0)
                
            _traj = np.concatenate([np.tile(traj_flat, (len(traj_fut), 1, 1)), traj_fut], axis=1)
            traj_base.append(_traj)
        traj_base = np.concatenate(traj_base, axis=0)
        
        traj = np.tile(traj_base, (int(N_seqs / traj_base.shape[0]), 1, 1))
        # traj[:, -pred_len:] += np.random.normal(loc=np.zeros_like(traj[:, -pred_len:]), scale=np.ones_like(traj[:, -pred_len:]) * std)
        # traj += np.random.normal(loc=np.zeros_like(traj), scale=np.ones_like(traj) * std)
        shape = np.array(traj.shape)

        # linear_scale
        if False:
            shape[1] = 1
            scale = np.arange(1., pred_len+1., 1., dtype=float)[None, :, None] / pred_len
            traj[:, -pred_len:] += np.random.normal(loc=np.zeros(shape), scale=np.ones(shape) * std) * scale
        # random whole
        if True:
            scale = np.ones(shape)[[0], -pred_len:] * std
            traj += np.random.normal(loc=np.zeros(shape), scale=np.ones(shape) * std)
        
        id_seq = np.tile(np.arange(N_seqs)[:, None, None], [1, seq_len, 1])
        traj = np.concatenate([id_seq, traj], axis=2)
        
        id_frame = np.arange(N_seqs * seq_len)[:, None]
        traj = np.concatenate([id_frame, traj.reshape([N_seqs * seq_len, -1])], axis=1)

        data = pd.DataFrame(traj, columns=['frame_id', 'track_id', 'pos_x', 'pos_y'])
        data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
        data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['track_id'].astype(str)

        data.sort_values('frame_id', inplace=True)
        max_timesteps = data['frame_id'].max()

        scene = Scene(timesteps=max_timesteps+1, dt=dt, name=f'{desired_source}_{data_class}', aug_func=None)
        scene.nodes = [preprocess_node(data, node_id, dt) for node_id in tqdm(pd.unique(data['node_id']))]
        print(scene)
        scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes
        
        if data_class == 'test':
            env.gt_dist_type = 'gmm'
            traj_base = traj_base.transpose(1, 0, 2)
            scale = scale.transpose(1, 0, 2)
            mix = D.Categorical(torch.ones(traj_base[-pred_len:].shape[:2]))
            comp = D.Independent(D.Normal(torch.tensor(traj_base[-pred_len:]), torch.tensor(scale * std)), 1)
            env.gt_dist = D.MixtureSameFamily(mix, comp)
            
            path = output_folder_name / 'visualize' / f'{desired_source}_{data_class}'
            visualize_env(env, path)
            
        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

"""
# Process simulated dataset
for desired_source in ['simunicycle']:
    for data_class in ['train', 'val', 'test']:
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = output_folder_name / ('_'.join([desired_source, data_class]) + '.pkl')

        d = {'train': 10000,
             'val': 1000,
             'test': 1000}
        N_seqs = d[data_class]
        obs_len = 8
        pred_len = 12
        seq_len = obs_len + pred_len
        
        max_pos_change = 0.1
        max_ang_acc = np.pi / 3 / pred_len
        
        traj_flat_x = np.arange(-obs_len+1, 1., 1., dtype=float)
        traj_flat_y = np.zeros_like(traj_flat_x, dtype=float)
        traj_flat = np.concatenate([traj_flat_x[:, None], traj_flat_y[:, None]], axis=1)
        
        traj_flat = np.tile(traj_flat[None], (int(N_seqs), 1, 1))
        
        max_ang_vel = np.arange(1, pred_len + 1) * max_ang_acc
        
        traj_flat_vel_x = np.ones(pred_len)
        traj_flat_vel_y = np.zeros(pred_len)
        traj_flat_vel = np.concatenate([traj_flat_vel_x[:, None], traj_flat_vel_y[:, None]], axis=1)
        traj_flat_vel = np.tile(traj_flat_vel[None], (int(N_seqs), 1, 1))
        
        traj_flat_vel[..., 0, 0] += np.random.uniform(low=-1.0, high=1.0, size=N_seqs) * max_pos_change
        
        deg = np.random.uniform(low=-1.0, high=1.0, size=N_seqs)[:, None] * max_ang_vel[None, :]
        c, s = np.cos(deg), np.sin(deg)
        upper = np.stack([c, s], axis=-1)
        lower = np.stack([-s, c], axis=-1)
        rotmat = np.stack([upper, lower], axis=-2)
        traj_fanshape_vel = np.einsum('ijkl,ijk->ijl', rotmat, traj_flat_vel)
        traj_fanshape = np.cumsum(traj_fanshape_vel, axis=1)
        
        traj = np.concatenate([traj_flat, traj_fanshape], axis=1)
        id_seq = np.tile(np.arange(N_seqs)[:, None, None], [1, seq_len, 1])
        traj = np.concatenate([id_seq, traj], axis=2)
        traj[..., 0] *= 0.7
        
        id_frame = np.arange(N_seqs * seq_len)[:, None]
        traj = np.concatenate([id_frame, traj.reshape([N_seqs * seq_len, -1])], axis=1) 

        data = pd.DataFrame(traj, columns=['frame_id', 'track_id', 'pos_x', 'pos_y'])
        data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
        data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['track_id'].astype(str)

        data.sort_values('frame_id', inplace=True)

        max_timesteps = data['frame_id'].max()

        scene = Scene(timesteps=max_timesteps+1, dt=dt, name=f'{desired_source}_{data_class}', aug_func=None)
        scene.nodes = [preprocess_node(data, node_id, dt) for node_id in tqdm(pd.unique(data['node_id']))]
        print(scene)
        scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')
        env.scenes = scenes
        
        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        
        if data_class == 'test':
            env.gt_dist_type = 'kde'
            data = torch.Tensor(traj_fanshape).cuda()
            kdes = []
            for i in range(pred_len):
                kdes.append(GaussianKDE(data[:, i]))
            env.gt_dist = kdes
            path = output_folder_name / 'visualize' / f'{desired_source}_{data_class}'
            visualize_env(env, path)
"""