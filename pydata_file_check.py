import json
json_path = r'E:\GeoVINS\AerialVL\vpr_training_data\val\images\CITY_7\info.json'
with open(json_path, 'r') as f:
    data = json.load(f)

pass


from scipy.io import loadmat
path = r'E:\GeoVINS\VPR\datasets\Pittsburgh\pitts30k_val.mat'
mat = loadmat(path)
matStruct = mat['dbStruct'].item()
numDb = matStruct[5].item()
pass

l = [[] for _ in range(8)]
pass


import torch
import numpy as np
a = np.load('./datasets/Pittsburgh/pitts30k_test_gt.npy', allow_pickle=True)
pass