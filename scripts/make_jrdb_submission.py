import os
import sys
import argparse
from pathlib import Path

from yacs.config import CfgNode
from tqdm import tqdm
import numpy as np
import torch

sys.path.append('./')
from keepflow.utils import load_config
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics

sys.path.append('extern/traj')
from jrdb_traj.jrdb_baselines.trajnetbaselines.lstm.tools.reader import Reader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch testing code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--device", type=str, default="cuda:0")
    
    return parser.parse_args()


def make_jrdb_submission(cfg: CfgNode):
    data_loader = build_dataloader(cfg, rand=False, split="test")
    model = build_model(cfg)
    metrics = build_metrics(cfg)
    try:
        model.load()
    except FileNotFoundError:
        print("no model saved")
        
    model.eval()
        
    idx_list = []
    pred_list = []
    for data_dict in tqdm(data_loader, leave=False):
            data_dict = {k: data_dict[k].to(cfg.DEVICE) 
                        if isinstance(data_dict[k], torch.Tensor)
                        else data_dict[k]
                        for k in data_dict}
            with torch.no_grad():
                result_dict = model.predict(data_dict, return_prob=False)
            
            result_dict = metrics.denormalize([result_dict])[0]
            idx_list += result_dict['index']
            pred_list.append(result_dict[('pred', 0)])
            
    pred_list = torch.cat(pred_list)
    
    dataset_index = 0
    path_dataset = Path('extern/traj/jrdb_traj/jrdb_baselines/DATA_BLOCK/jrdb_traj/test')
    path_save = Path(cfg.SAVE_DIR) / 'jrdb_submission' / 'data'
    os.makedirs(path_save, exist_ok=True)
    for p in path_save.iterdir():
        p.unlink()
    for p in path_dataset.iterdir():
        scenes = load_test_datasets(p)
        
        write_predictions(pred_list, idx_list, scenes, cfg, p.stem, dataset_index, path_save)
        dataset_index += 1

    
def load_test_datasets(path):
    print('Dataset Name: ', path.stem)
    reader = Reader(path, scene_type='paths')
    scenes = [(path.stem, s_id, s) for s_id, s in reader.scenes()]

    return scenes


def write_predictions(pred_list, idx_list, scenes, cfg, dataset_name, dataset_index, path_save):
    """Write predictions corresponding to the scenes in the respective file"""
    model_name = cfg.MODEL.TYPE
    
    ## Write All Predictions
    data = []
    for _, scene_id, paths in scenes:
        matched = [dataset_name == idx[0] and scene_id == int(idx[1]) for idx in idx_list]
        
        observed_path = paths[0]
        frame_diff = observed_path[1].frame - observed_path[0].frame
        first_frame = observed_path[9-1].frame + frame_diff
        ped_id = observed_path[0].pedestrian
        
        for idx, pred in zip(np.array(idx_list)[matched], pred_list[matched]):
            for frame in range(len(pred)):
                data.append([first_frame + frame * frame_diff, ped_id, pred[frame, 0].item(), pred[frame, 1].item(), 1.0])
                
    data = np.array(data)
    with open(path_save / (str(dataset_index).zfill(4)+'.txt'), 'a') as txtfile:
        for pred_id in range(12):
            for row_id in range(data.shape[0]):
                if data[row_id,0] == data[pred_id,0]:

                    data_final = [int(data[row_id,0]),int(data[row_id,1]), 'Pedestrian', 0, 0, -1 ,-1, -1, -1, -1, data[row_id,2], data[row_id,3]]

                    for d_final in data_final:
                        txtfile.write(str(d_final))
                        txtfile.write(' ')
                    txtfile.write('\n')
        

def main() -> None:
    args = parse_args()
    assert os.path.basename(args.config_file) == 'jrdb.yml'
    
    cfg = load_config(args)

    make_jrdb_submission(cfg)
    
if __name__ == '__main__':
    main()