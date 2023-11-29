__author__ = 'jianjin'

import numpy as np
import os
from PIL import Image
import tensorflow as tf
import logging
import random

logger = logging.getLogger(__name__)


class InputHandle:
    def __init__(self, datas, indices, configs, transform):
        self.name = configs['name'] + ' iterator'
        self.image_height = configs['image_height']
        self.image_width = configs['image_width']
        self.datas = datas
        self.indices = indices
        self.input_n = configs['input_n']
        self.output_n = configs['output_n']
        self.seq_len = self.input_n + self.output_n
        self.injection_action = configs['injection_action']
        
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        begin = self.indices[index][-1]
        end = begin + self.seq_len
        ret_seq = np.zeros(
            (self.seq_len, self.image_height, self.image_width, 7)).astype(np.float32)
        k = 0
        for serialized_example in tf.compat.v1.python_io.tf_record_iterator(self.indices[index][0]):
            if k == self.indices[index][1]:
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                break
            k += 1
        for j in range(begin, end):
            action_name = str(j) + '/action'
            action_value = np.array(example.features.feature[action_name].float_list.value)
            if action_value.shape == (0,):  # End of frames/data
                print("error! " + str(self.indices[index]))
            ret_seq[j - begin, :, :, 3:] = np.stack([np.ones([64, 64]) * i for i in action_value], axis=2)

            # endeffector_pos_name = str(j) + '/endeffector_pos'
            # endeffector_pos_value = list(example.features.feature[endeffector_pos_name].float_list.value)
            # endeffector_positions = np.vstack((endeffector_positions, endeffector_pos_value))

            aux1_image_name = str(j) + '/image_aux1/encoded'
            aux1_byte_str = example.features.feature[aux1_image_name].bytes_list.value[0]
            aux1_img = Image.frombytes('RGB', (64, 64), aux1_byte_str)
            aux1_arr = np.array(aux1_img.getdata()).reshape((aux1_img.size[1], aux1_img.size[0], 3))

            # main_image_name = str(j) + '/image_main/encoded'
            # main_byte_str = example.features.feature[main_image_name].bytes_list.value[0]
            # main_img = Image.frombytes('RGB', (64, 64), main_byte_str)
            # main_arr = np.array(main_img.getdata()).reshape((main_img.size[1], main_img.size[0], 3))

            ret_seq[j - begin, :, :, :3] = aux1_arr.reshape(64, 64, 3) / 255
            
        ret_seq = self.transform(ret_seq)
            
        return [index, ret_seq[:self.input_n], ret_seq[self.input_n:self.input_n + self.output_n]]


class DataProcess:
    def __init__(self, configs):
        self.configs = configs
        self.train_data_path = configs['train_data_path']
        self.valid_data_path = configs['valid_data_path']
        self.image_height = configs['image_height']
        self.image_width = configs['image_width']
        self.input_n = configs['input_n']
        self.output_n = configs['output_n']
        self.seq_len = self.input_n + self.output_n

    def load_data(self, path, mode='train'):
        path = os.path.join(path, 'softmotion30_44k')
        if mode == 'train':
            path = os.path.join(path, 'train')
        elif mode == 'test':
            path = os.path.join(path, 'test')
        else:
            print("ERROR!")
        print('begin load data' + str(path))

        video_fullpaths = []
        indices = []

        tfrecords = os.listdir(path)
        tfrecords.sort()
        num_pictures = 0

        for tfrecord in tfrecords:
            filepath = os.path.join(path, tfrecord)
            video_fullpaths.append(filepath)
            k = 0
            for serialized_example in tf.compat.v1.python_io.tf_record_iterator(os.path.join(path, tfrecord)):
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                i = 0
                while True:
                    action_name = str(i) + '/action'
                    action_value = np.array(example.features.feature[action_name].float_list.value)
                    if action_value.shape == (0,):  # End of frames/data
                        break
                    i += 1
                num_pictures += i
                for j in range(i - self.seq_len + 1):
                    indices.append((filepath, k, j))
                k += 1
        print("there are " + str(num_pictures) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return video_fullpaths, indices

    def get_train_input_handle(self, transform):
        train_data, train_indices = self.load_data(self.train_data_path, mode='train')
        return InputHandle(train_data, train_indices, self.configs, transform)

    def get_test_input_handle(self, transform):
        test_data, test_indices = self.load_data(self.valid_data_path, mode='test')
        return InputHandle(test_data, test_indices, self.configs, transform)
