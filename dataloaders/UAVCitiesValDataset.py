from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat

import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
import json

root_dir = r'E:\GeoVINS\AerialVL\vpr_training_data\val\images'

class ValSetInfo():
    def __init__(self):
        self.dataset = None
        self.dataset_root = None
        self.numDb = None
        self.numQ = None
        self.dbImage = None
        self.qImage = None
        self.PosImageGT = None

def input_transform(image_size=None):
    return T.Compose([
        T.Resize(image_size),# interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])




def parse_info_json(json_path):
    with open(json_path, 'r') as f:
        struct_info = json.load(f)

    dataset_info = ValSetInfo()
    dataset_info.dataset = struct_info['dataset']
    dataset_info.dataset_root = struct_info['dataset_root']
    dataset_info.numDb = struct_info['numDb']
    dataset_info.numQ = struct_info['numQ']
    dataset_info.dbImage = struct_info['dbImage']
    dataset_info.qImage = struct_info['qImage']
    dataset_info.PosImageGT = struct_info['PosImageGT']

    return dataset_info

def get_whole_dataset(dataset_name, input_transform):
    json_path = root_dir + '\\' + dataset_name + r'\info.json'
    return UAVCitiesValDataset(json_path, input_transform=input_transform)

class UAVCitiesValDataset(Dataset):
    def __init__(self, info_json_path, input_transform=None):
        super().__init__()
        self.input_transform = input_transform
        self.dataset_info = parse_info_json(info_json_path)
        dataset_root = root_dir + self.dataset_info.dataset
        self.images = [join(dataset_root, dbImg) for dbImg in self.dataset_info.dbImage]
        self.images += [join(dataset_root, qImg) for qImg in self.dataset_info.qImage]
        self.positives = None

    def __getitem__(self, index):

        try:
            img = Image.open(self.images[index])
        except UnidentifiedImageError:
            print(f'Image {self.images[index]} could not be loaded')
            img = Image.new('RGB', (224, 224))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index
        
    def __len__(self):
        return len(self.images)

    def getPositives(self):
        positives_path = self.dataset_info.PosImageGT
        Positives = np.array([], dtype=object)
        for i in range(len(positives_path)):
            positives_line = np.array([], dtype=np.int64)
            for j in range(len(positives_path[i])):
                qImg_idx = self.dataset_info.qImage.index(positives_path[i][j])
                positives_line = np.append(positives_line, qImg_idx)
        Positives = np.append(Positives, positives_line)
        self.positives = Positives
        return self.positives
