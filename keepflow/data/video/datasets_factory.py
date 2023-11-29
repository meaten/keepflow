import numpy as np
import torch
from data.VP import kth_action, mnist, bair

datasets_map = {
    'mnist': mnist,
    'kth': kth_action,
    'bair': bair,
}


def VP_dataset(dataset_name, data_path, split,
               img_width, input_n, output_n, injection_action):
    
    if split == "train":
        transform = TrainTransform()
    elif split == "test":        
        transform = TestTransform()
    else:
        raise ValueError
    
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    if dataset_name == 'mnist':
        input_param = {'path': data_path,
                       'input_data_type': 'float32',
                       'input_n': input_n,
                       'output_n': output_n,
                       'name': dataset_name + ' iterator'}
        data_loader = datasets_map[dataset_name].InputHandle(input_param, transform)
        return data_loader
        

    if dataset_name == 'kth':
        input_param = {'path': data_path,
                       'image_width': img_width,
                       'input_n': input_n,
                       'output_n': output_n,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if split == "train":
            data_loader = input_handle.get_train_input_handle(transform)
        elif split == "test":
            data_loader = input_handle.get_test_input_handle(transform)
        else:
            raise ValueError

    if dataset_name == 'bair':
        test_input_param = {'valid_data_path': data_path,
                            'train_data_path': data_path,
                            'image_width': img_width,
                            'image_height': img_width,
                            'input_n': input_n,
                            'output_n': output_n,
                            'injection_action': injection_action,
                            'input_data_type': 'float32',
                            'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(test_input_param)
        if split == "train":
            data_loader = input_handle.get_train_input_handle(transform)
        elif split == "test":
            data_loader = input_handle.get_test_input_handle(transform)
        else:
            raise ValueError

        
    return data_loader

class TrainTransform:
    def __init__(self):
        self.augment = Compose([
            RandomMirror()
        ])

    def __call__(self, video):
        video = self.augment(video)
        return video

class TestTransform:
    def __init__(self):
        self.augment = Compose([
            DoNothing()
        ])

    def __call__(self, video):
        video = self.augment(video)
        return video


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video):
        for t in self.transforms:
            video = t(video)

        return video
    
class RandomMirror:
    def __call__(self, video):
        if np.random.randint(2):
            video = np.flip(video, axis=0).copy()

        return video
    
class DoNothing:
    def __call__(self, video):
        return video
    
