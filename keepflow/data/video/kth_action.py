__author__ = 'gaozhifeng'
import numpy as np
import os
import cv2
from PIL import Image
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class InputHandle(DataLoader):
    def __init__(self, datas, indices, input_param, transform):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.input_n = input_param["input_n"]
        self.output_n = input_param["output_n"]
        self.seq_len = self.input_n + self.output_n
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        begin = self.indices[index]
        end = begin + self.seq_len
        data_slice = self.datas[begin:end, :, :, :]
        
        data_slice = self.transform(data_slice)
        
        return [index, data_slice[:self.input_n], data_slice[self.input_n:self.input_n+self.output_n]]


class DataProcess:
    def __init__(self, input_param):
        self.path = input_param['path']
        self.category_1 = ['boxing', 'handclapping', 'handwaving', 'walking']
        self.category_2 = ['jogging', 'running']
        self.category = self.category_1 + self.category_2
        self.image_width = input_param['image_width']

        self.train_person = ['01', '02', '03', '04', '05', '06', '07', '08',
                             '09', '10', '11', '12', '13', '14', '15', '16']
        self.test_person = ['17', '18', '19', '20', '21', '22', '23', '24', '25']

        self.input_param = input_param
        self.input_n = input_param['input_n']
        self.output_n = input_param['output_n']
        self.seq_len = self.input_n + self.output_n

    def load_data(self, path, mode='train'):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        if mode == 'train':
            person_id = self.train_person
        elif mode == 'test':
            person_id = self.test_person
        else:
            print("ERROR!")
        print('begin load data' + str(path))

        frames_np = []
        frames_file_name = []
        frames_person_mark = []
        frames_category = []
        person_mark = 0

        c_dir_list = self.category
        frame_category_flag = -1
        for c_dir in c_dir_list: # handwaving
            if c_dir in self.category_1:
                frame_category_flag = 1 # 20 step
            elif c_dir in self.category_2:
                frame_category_flag = 2 # 3 step
            else:
                print("category error!!!")

            c_dir_path = os.path.join(path, c_dir)
            p_c_dir_list = os.listdir(c_dir_path)

            for p_c_dir in p_c_dir_list: 
                if p_c_dir[6:8] not in person_id:
                    continue
                person_mark += 1
                dir_path = os.path.join(c_dir_path, p_c_dir)
                filelist = os.listdir(dir_path)
                filelist.sort() 
                for file in filelist: 
                    if file.startswith('image') == False:
                        continue
                    # print(file)
                    # print(os.path.join(dir_path, file))
                    frame_im = Image.open(os.path.join(dir_path, file))
                    frame_np = np.array(frame_im)  # (1000, 1000) numpy array
                    # print(frame_np.shape)
                    frame_np = frame_np[:, :, 0] #
                    frames_np.append(frame_np)
                    frames_file_name.append(file)
                    frames_person_mark.append(person_mark)
                    frames_category.append(frame_category_flag)
        # is it a begin index of sequence
        indices = []
        index = len(frames_person_mark) - 1
        while index >= self.seq_len - 1:
            if frames_person_mark[index] == frames_person_mark[index - self.seq_len + 1]:
                end = int(frames_file_name[index][6:10])
                start = int(frames_file_name[index - self.seq_len + 1][6:10])
                # TODO: mode == 'test'
                if end - start == self.seq_len - 1:
                    indices.append(index - self.seq_len + 1)
                    if frames_category[index] == 1:
                        index -= self.seq_len - 1
                    elif frames_category[index] == 2:
                        index -= 2
                    else:
                        print("category error 2 !!!")
            index -= 1

        frames_np = np.asarray(frames_np)
        data = np.zeros((frames_np.shape[0], self.image_width, self.image_width , 1))
        for i in range(len(frames_np)):
            temp = np.float32(frames_np[i, :, :])
            data[i,:,:,0]=cv2.resize(temp,(self.image_width,self.image_width))/255
        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return data, indices

    def get_train_input_handle(self, transform):
        train_data, train_indices = self.load_data(self.path, mode='train')
        return InputHandle(train_data, train_indices, self.input_param, transform)

    def get_test_input_handle(self, transform):
        test_data, test_indices = self.load_data(self.path, mode='test')
        return InputHandle(test_data, test_indices, self.input_param, transform)

