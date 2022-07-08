from torch.utils.data import Dataset
import numpy as np
import torch
from copy import deepcopy
from data.MP import data_utils


def seq_collate(data):
    index, obs, decoder_input, gt, gt_all, mean, std, dim_used = zip(*data)

    obs = normalize_data(obs)
    decoder_input = normalize_data(decoder_input)
    gt = normalize_data(gt)
    gt_all = normalize_data(gt_all)

    return {
        "index": torch.IntTensor(index),
        "obs": obs,
        "decoder_input": decoder_input,
        "gt": gt,
        "gt_all": gt_all,
        "mean": torch.Tensor(mean[0]),
        "std": torch.Tensor(std[0]),
        "dim_used": torch.Tensor(dim_used[0])
    }

def normalize_data(sequence):
    # (bs, T, ...) -> (T, bs, ...)
    sequence = torch.Tensor(np.array(sequence))
    sequence.transpose_(0, 1)

    return sequence

class H36motion(Dataset):

    def __init__(self, path_to_data, actions, input_n=10, output_n=10, split=0, sample_rate=2, data_mean=0,
                 data_std=0, onehot=True, onehotencoder=None, load_3d=False):
        
        self.path_to_data = path_to_data
        self.split = split
        self.load_3d = load_3d
        subs = [[1, 6, 7, 8, 9], [5], [11]]

        acts = data_utils.define_actions(actions)

        # subs = np.array([[1], [5], [11]])
        # acts = ['walking']

        subjs = subs[split]
        if not self.load_3d:
            all_seqs, dim_ignore, dim_use, data_mean_, data_std_, action_list = data_utils.load_data(path_to_data, subjs, acts,
                                                                                  sample_rate,
                                                                                  input_n + output_n,
                                                                                  data_mean=data_mean,
                                                                                  data_std=data_std,
                                                                                  input_n=input_n)
        else:
            all_seqs, dim_ignore, dim_use, data_mean_, data_std_, action_list = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                 input_n + output_n)

        self.data_mean = data_mean if split == 1 else data_mean_
        self.data_std = data_std if split == 1 else data_std_
        
        self.input_n = input_n
        self.output_n = output_n
        self.seq_len = input_n + output_n
        
        # first 6 elements are global translation and global rotation
        # dim_used = np.where(np.asarray(dim_use) > 5)[0]
        self.dim_used = dim_use
        self.all_seqs = all_seqs

        #self.seqs = (all_seqs - self.data_mean) / self.data_std
        self.seqs = deepcopy(all_seqs)
        self.data_mean = np.zeros_like(self.data_mean)
        self.data_std = np.zeros_like(self.data_std) + 1
        self.seqs = self.seqs[:, :, self.dim_used]
        self.onehotencoder = onehotencoder
        if onehot:
            if self.onehotencoder is None:
                from sklearn.preprocessing import OneHotEncoder
                self.onehotencoder = OneHotEncoder()
                action_list_unique = np.sort(np.unique(action_list))[:, None]
                self.onehotencoder.fit(action_list_unique)
            onehot = self.onehotencoder.transform(np.array(action_list)[:, None])
            self.seqs = np.concatenate([self.seqs, np.tile(onehot.toarray()[:, None, :], [1, input_n + output_n, 1])], axis=2)
            
    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        all_seq = self.all_seqs[item]
        seq = self.seqs[item]

        # padding the observed sequence so that it has the same length as observed + future sequence
        return [index,
                seq[:self.input_n, :], seq[self.input_n-1:-1, :],
                seq[self.input_n:, :], all_seq[self.input_n:, :],
                self.data_mean, self.data_std, self.dim_used]
        """ 
        self.dct_n = dct_n
        self.dct_m, _ = data_utils.get_dct_matrix(self.seq_len)
        self.dct_m = self.dct_m[:dct_n]

        # for dct
        pad_idx = np.repeat([self.input_n - 1], self.output_n)
        i_idx = np.append(np.arange(0, self.input_n), pad_idx)
        input_dct_seq = np.matmul(self.dct_m, seq[i_idx, :])
        input_dct_seq = input_dct_seq.transpose()

        output_dct_seq = np.matmul(self.dct_m, seq)
        output_dct_seq = output_dct_seq.transpose()

        return input_dct_seq, output_dct_seq, seq_return
        """


