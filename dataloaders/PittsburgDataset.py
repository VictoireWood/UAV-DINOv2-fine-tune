from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat

import torchvision.transforms as T
import torch.utils.data as data


from PIL import Image, UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors

root_dir = '../data/Pittsburgh/'

if not exists(root_dir):
    raise FileNotFoundError(
        'root_dir is hardcoded, please adjust to point to Pittsburgh dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')


def input_transform(image_size=None):
    return T.Compose([
        T.Resize(image_size),# interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



def get_whole_val_set(input_transform):
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)


def get_250k_val_set(input_transform):
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)


def get_whole_test_set(input_transform):
    structFile = join(struct_dir, 'pitts30k_test.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)


def get_250k_test_set(input_transform):
    structFile = join(struct_dir, 'pitts250k_test.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)

def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return WholeDatasetFromStruct(structFile,
                                  input_transform=input_transform(),
                                  onlyDB=onlyDB)

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
                    utmQ, numDb, numQ, posDistThr,
                    posDistSqThr, nonTrivPosDistSqThr)
# NOTE - <https://github.com/Nanne/pytorch-NetVlad/issues/26>
# utm is the Universal Transverse Mercator coordinate system, i.e., it tells you where in Tokyo the images are taken.
# PosDisThr is the positive distance threshold, Sq is the squared version on that, 
# and nonTrivPosDisSqThr is the non trivial positive distance threshold. 
# These thresholds are in meters and are the various distances used to determine which images are positives or potential positives.


class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm)
                            for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

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
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            # NOTE - 最近邻检索<https://blog.csdn.net/lovego123/article/details/67638789>
            knn = NearestNeighbors(n_jobs=-1)   # NOTE - 近邻搜索的并行度，默认为None，表示1；-1表示使用所有cpu
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                                  radius=self.dbStruct.posDistThr)
            # NOTE - 找到Database的照片的各个拍摄位置的最近的Query的拍摄点位和距离
            # 返回最近Query到Database各个照片拍摄位置的距离和点位索引

        return self.positives
