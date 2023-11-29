import numpy as np
import torch
from torch.utils.data import DataLoader

class InputHandle(DataLoader):
    def __init__(self, input_param, transform):
        self.path = input_param['path']
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.input_n = input_param["input_n"]
        self.output_n = input_param["output_n"]
        self.data = {}
        self.load()
        
        self.transform = transform

    def load(self):
        dat_1 = np.load(self.path)
        for key in dat_1.keys():
            self.data[key] = dat_1[key]
        for key in self.data.keys():
            print(key)
            print(self.data[key].shape)

    def __len__(self):
        return self.data['clips'].shape[1]
    
    def __getitem__(self, index):
        """
        begin = self.data["clips"][0, index, 0]
        end = self.data["clips"][0, index, 0] + self.data["clips"][0, index, 1]
        input_slice = self.data["input_raw_data"][begin:end, :, :, :]
        input_slice = np.transpose(input_slice,(0,2,3,1))
        begin = self.data["clips"][1, index, 0]
        end = self.data["clips"][1, index, 0] + self.data["clips"][1, index, 1]
        output_slice = self.data["input_raw_data"][begin:end, :, :, :]
        output_slice = np.transpose(output_slice,(0,2,3,1))
        """
        begin = self.data["clips"][0, index, 0]
        end = self.data["clips"][1, index, 0] + self.data["clips"][1, index, 1]
        data_slice = self.data["input_raw_data"][begin:end, :, :, :]
        data_slice = np.transpose(data_slice,(0,2,3,1))
        
        data_slice = self.transform(data_slice)
        
        return [index, data_slice[:self.input_n], data_slice[self.input_n:self.input_n+self.output_n]]


def seq_collate(data):
    (index_list, obs_seq_list, gt_seq_list) = zip(*data)
    
    obs = np.array(obs_seq_list, dtype=np.float32)
    gt = np.array(gt_seq_list, dtype=np.float32)

    obs = torch.from_numpy(obs).permute(1, 0, 4, 2, 3)
    gt = torch.from_numpy(gt).permute(1, 0, 4, 2, 3)
    out = {
        "index": torch.IntTensor(index_list),
        "obs": obs,
        "gt": gt}

    return out