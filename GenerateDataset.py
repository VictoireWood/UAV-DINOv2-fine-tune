'''
有个致命的问题，由于randint的范围一直是整个能取的范围，所以不确定会不会有两轮循环取出来的两组图片位置距离过近，实际上可以合并。
有个办法就是每取出一组图片就把可能重复的范围从原来的范围中挖去，把剩余可取的范围做成一个列表，下次randint的时候在列表中随机取一个区间进行randint。
这个功能比较麻烦，最后再加。

不行，取不满足够的图片，应该去掉的是x和y横竖两条的并集而不是交集。
记录所有随机取到的标准中心点，用最近邻搜索来删掉后来取得太近的。
如果太近就重新取，保证能取满1000个。
'''

import os
import sys
import shutil
from glob import glob

import random
from tqdm import tqdm
import pandas as pd
import math
from os.path import join
import numpy as np
from itertools import product
import cv2
import json
from pyproj import Transformer
from haversine import haversine, Unit
from sklearn.neighbors import NearestNeighbors


CITIES = [
    'CITY_1',
    'CITY_2',
    'CITY_3',
    'CITY_4',
    'CITY_5',
    'CITY_6',
    'CITY_7'
]

boundary = 5

TRAIN_CITIES = CITIES[:boundary]
VAL_CITIES = CITIES[boundary:]

def create_noexist_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

BASE_DIR = r'E:\GeoVINS\AerialVL\vpr_training_data'
map_database_path = BASE_DIR + r'\raw_satellite_imageries\*'
attach_1st = [r'\train', r'\val']
attach_2nd = [r'\dataframes', r'\images']
attach_3rd = [r'\database', r'\query']
SET_PATH = []
IMG_PATH = [[], []]
DB_Q_VAL_PATH =[[], []]

for i in range(2):
    base_tmp = []
    base_1st = BASE_DIR + attach_1st[i]
    for j in range(2):
        base_2nd = base_1st + attach_2nd[j]
        create_noexist_dir(base_2nd)
        base_tmp.append(base_2nd)
    SET_PATH.append(base_tmp)
    
for i in range(len(CITIES)):
    city = CITIES[i]
    if city in TRAIN_CITIES:
        city_img_path = SET_PATH[0][1] + '\\' + city
        create_noexist_dir(city_img_path)
        IMG_PATH[0].append(city_img_path)
    else:
        city_img_path = SET_PATH[1][1] + '\\' + city
        IMG_PATH[1].append(city_img_path)
        db_path = city_img_path + attach_3rd[0]
        q_path = city_img_path + attach_3rd[1]
        create_noexist_dir(db_path)
        create_noexist_dir(q_path)
        DB_Q_VAL_PATH[0].append(db_path)
        DB_Q_VAL_PATH[1].append(q_path)
VAL_ROOT = SET_PATH[1][1]   # 到val/images的那个文件夹
del city, city_img_path, i, j, db_path, q_path, base_tmp, base_1st, base_2nd

class dbImgPosGT:
    def __init__(self, dbImagePath, posQImagePath):
        self.dbImagePath = dbImagePath
        self.posQImagePath = posQImagePath



def lat_lon_to_utm(lat, lon):   # 存疑
    Zone = np.round((183 + lon)/6,0)
    print("Zone is ", Zone)
    EPSG=32700 - np.round((45 + lat)/90, 0) * 100 + np.round((183 + lon)/6, 0)
    EPSG = int(EPSG)
    print("EPSG is ", EPSG)


def generate_database_images():
    # important parameters
    map_w = 5000    # 原始图像的像素宽
    map_h = 5000

    img_w = 800     # 目标图像所在区域在原始图像上所占的像素宽和高
    img_h = 800

    target_w = 500  # 目标图像的像素宽（需要resize）
    target_h = 500

    shift_range = 50

    # map_database_path = '/home/cloudam/Dataset/JimoDataset/dataset/*'
    map_database_path = r'E:\GeoVINS\AerialVL\vpr_training_data\raw_satellite_imageries\*'
    citys_path = glob(map_database_path)
    for city_path in citys_path:
        map_list = glob(os.path.join(city_path, '*.png'))
        map_number = len(map_list)
        query_map_idx_max = math.floor(map_number/2.0)
        # map_ids = product(map_list, repeat=2)
        for query_img_idx in range(query_map_idx_max):
            map1 = cv2.imread(map_list[query_img_idx])
            database_img_idx = query_img_idx + query_map_idx_max if (query_img_idx + query_map_idx_max) < map_number else map_number-1
            map2 = cv2.imread(map_list[database_img_idx])

            for generate_loop in range(1000):
                if map1.shape[0] != map_w or map1.shape[1] != map_h:
                    print("Wrong Map Shape!")
                    return 0
                else:
                    lon_lat_path = os.path.join(city_path, 'lon_lat.txt')

                    with open(lon_lat_path, "r") as f:
                        lines = f.readlines()
                        lat_n = float((lines[0].split(' '))[0])
                        lat_s = float((lines[0].split(' '))[1])
                        lat_res = (lat_s - lat_n) / 5000

                        lon_e = float((lines[1].split(' '))[0])
                        lon_w = float((lines[1].split(' '))[1])
                        lon_res = (lon_e - lon_w) / 5000

                        f.close()

                    loc1_x = random.randint(0, map_w - img_w - shift_range)
                    loc1_y = random.randint(0, map_h - img_h - shift_range)

                    shift_x = random.randint(0, shift_range)
                    shift_y = random.randint(0, shift_range)

                    loc2_x = loc1_x + shift_x
                    loc2_y = loc1_y + shift_y

                    cur1_lon = str((loc1_x + img_w / 2) * lon_res + lon_w)
                    cur1_lat = str((loc1_y + img_h / 2) * lat_res + lat_s)

                    cur2_lon = str((loc2_x + img_w / 2) * lon_res + lon_w)
                    cur2_lat = str((loc2_y + img_h / 2) * lat_res + lat_s)

                    # 邵星雨改（所有log改成lon的地方都是）
                    cur1_lon_LT = str((loc1_x) * lon_res + lon_w)
                    cur1_lat_LT = str((loc1_y) * lat_res + lat_s)
                    cur1_lon_RB = str((loc1_x + img_w) * lon_res + lon_w)
                    cur1_lat_RB = str((loc1_y + img_h) * lat_res + lat_s)

                    cur2_lon_LT = str((loc2_x) * lon_res + lon_w)
                    cur2_lat_LT = str((loc2_y) * lat_res + lat_s)
                    cur2_lon_RB = str((loc2_x + img_w) * lon_res + lon_w)
                    cur2_lat_RB = str((loc2_y + img_h) * lat_res + lat_s)


                    img1 = map1[loc1_y:loc1_y + img_h, loc1_x:loc1_x + img_w]
                    img2 = map2[loc2_y:loc2_y + img_h, loc2_x:loc2_x + img_w]

                    img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)

                    cv2.imwrite('E:\\GeoVINS\\AerialVL\\vpr_training_data\\images\\map_database\\@map%s.png'
                                # '/home/cloudam/Dataset/JimoDataset/images/train/database/%s.png'
                                % ('@' + cur1_lon_LT + '@' + cur1_lat_LT + '@' + cur1_lon_RB + '@' + cur1_lat_RB + '@'), img1)
                    cv2.imwrite('E:\\GeoVINS\\AerialVL\\vpr_training_data\\images\\query_images\\@img%s.png'
                                # '/home/cloudam/Dataset/JimoDataset/images/train/queries/%s.png'
                                % ('@' + cur2_lon_LT + '@' + cur2_lat_LT + '@' + cur2_lon_RB + '@' + cur2_lat_RB + '@'), img2)
                    print("point one: ", loc1_x, loc1_y)
                    print("point two: ", loc2_x, loc2_y)
                    # print('Finish %s.png' % ('@' + cur1_log + '@' + cur1_lat + '@'))

def generate_train_dataset(train_set_generate = False):
    # important parameters
    map_w = 5000    # TODO - 原始图像的像素宽
    map_h = 5000

    img_w = 800     # TODO - 目标图像所在区域在原始图像上所占的像素宽和高
    img_h = 800

    target_w = 500  # TODO - 目标图像的像素宽（需要resize）
    target_h = 500

    shift_range = 50    # TODO - 两个接近相同的区域像差像素的多少（query和dataset之间）

    train_set = 1000
    val_set = 300

    max_place_w = map_w // (shift_range * 2)
    max_place_h = map_h // (shift_range * 2)
    max_place = max_place_w * max_place_h

    # NOTE - 分开两部分train和val，需要给各个城市的每张图设置一个特有的id，将数据记入csv
    
    cities_path = glob(map_database_path)
    header = pd.DataFrame(columns=['place_id', 'city_id', 'map_idx', 'lon', 'lat', 'lon_LT', 'lat_LT', 'lon_RB', 'lat_RB'])

    for city_path in cities_path:
        city_id = city_path.split('\\')[-1]
        if train_set_generate == True and city_id in TRAIN_CITIES:
            row_idx = 0
            pass
        # elif val_set_generate == True and city_id in VAL_CITIES:
        #     row_idx = 1
        #     pass
        else:
            continue

        map_list = glob(os.path.join(city_path, '*.png'))
        map_number = len(map_list)
        for check_idx in map_list:
            check_map = cv2.imread(check_idx)
            if check_map.shape[0] != map_w or check_map.shape[1] != map_h:
                sys.exit('Wrong map shape!')
            else:
                pass
        del check_idx, check_map
        
        # print(header)
        csv_dataframe = SET_PATH[row_idx][0] + '\\' + city_id + '.csv'
        header.to_csv(csv_dataframe, mode='w', index=False, header=True)
        if map_number < 4:
            completion_number = 4 - map_number
            choose_substitute = random.sample(map_list, completion_number)  # 补齐至少4个地图，要求文件中至少有两个地图
            for substitute in choose_substitute:
                substitute_cv = cv2.imread(substitute)
                substitute_name = substitute.replace('.png','_sub.png')
                cv2.imwrite(substitute_name, substitute_cv)
            map_list = glob(os.path.join(city_path, '*.png'))
            map_number = len(map_list)
        
        lon_lat_path = os.path.join(city_path, 'lon_lat.txt')
        lon_lat_range_np = np.loadtxt(lon_lat_path, delimiter=' ', dtype=np.float64)
        lon_lat_range = lon_lat_range_np.tolist()
        lat_n = lon_lat_range[0][0]
        lat_s = lon_lat_range[0][1]
        lat_res = (lat_s - lat_n) / 5000    # NOTE - 计算每一个像素宽度对应多少纬度变化
        lon_e = lon_lat_range[1][0]
        lon_w = lon_lat_range[1][1]
        lon_res = (lon_e - lon_w) / 5000    # NOTE - 计算每一个像素宽度对应多少经度变

        taken_points = []
        for generate_loop in range(1000):
            place_id = generate_loop
            std_loc_x = random.randint(0, map_w - img_w - shift_range)
            std_loc_y = random.randint(0, map_h - img_h - shift_range)
            while True:
                std_loc_x = random.randint(0, map_w - img_w - shift_range)
                std_loc_y = random.randint(0, map_h - img_h - shift_range)
                if len(taken_points) == 0:
                    break
                else:
                    neigh = NearestNeighbors(n_neighbors=1, radius=shift_range)
                    neigh.fit(taken_points)
                    idx = neigh.radius_neighbors([[std_loc_x, std_loc_y]], return_distance=False)
                    number = idx.item().size
                if number == 0:
                    break
            

            random.shuffle(map_list)
            for map_idx in range(map_number):
                if map_idx == 0:
                    loc_x = std_loc_x
                    loc_y = std_loc_y
                else:
                    shift_x = random.randint(0, shift_range)
                    shift_y = random.randint(0, shift_range)
                    loc_x = std_loc_x + shift_x
                    loc_y = std_loc_y + shift_y
                taken_points.append([loc_x, loc_y])
                map_origin = cv2.imread(map_list[map_idx])
                map_slice = map_origin[loc_y:loc_y + img_h, loc_x:loc_x + img_w]
                map_resize = cv2.resize(map_slice, (target_w, target_h), interpolation=cv2.INTER_AREA)
                lon_LT = ((loc_x) * lon_res + lon_w)
                lat_LT = ((loc_y) * lat_res + lat_s)
                lon_RB = ((loc_x + img_w) * lon_res + lon_w)
                lat_RB = ((loc_y + img_h) * lat_res + lat_s)
                lon = ((loc_x + img_w / 2) * lon_res + lon_w)
                lat = ((loc_y + img_h / 2) * lat_res + lat_s)
                data_line = pd.DataFrame([[place_id, city_id, map_idx, lon, lat, lon_LT, lat_LT, lon_RB, lat_RB]], columns=['place_id', 'city_id', 'map_idx','lon', 'lat', 'lon_LT', 'lat_LT', 'lon_RB', 'lat_RB'])
                data_line.to_csv(csv_dataframe, mode='a', index=False, header=False)
                if row_idx == 0:
                    col_idx = TRAIN_CITIES.index(city_id)
                # else:
                #     col_idx = VAL_CITIES.index(city_id)
                origin_img = map_list[map_idx].split('\\')[-1].replace('.png', '')
                cv2.imwrite(IMG_PATH[row_idx][col_idx] +f'\\@{city_id}@{str(place_id).zfill(4)}@{str(map_idx)}@{origin_img}@{str(lon)}@{str(lat)}@.png')
                # tmp = pd.read_csv(csv_dataframe)
                # print(tmp.iloc[-1])

def generate_validate_dataset(val_set_generate = False):
    # important parameters
    map_w = 5000    # TODO - 原始图像的像素宽
    map_h = 5000

    img_w = 800     # TODO - 目标图像所在区域在原始图像上所占的像素宽和高
    img_h = 800

    target_w = 500  # TODO - 目标图像的像素宽（需要resize）
    target_h = 500

    shift_range = 50    # TODO - 两个接近相同的区域像差像素的多少（query和dataset之间）

    train_set = 1000
    val_set = 300

    max_place_w = map_w // (shift_range * 2)
    max_place_h = map_h // (shift_range * 2)
    max_place = max_place_w * max_place_h

    dbImage = []
    qImage = []
    PosImageGT = []
    PosImageGT_idx = []
    numDb = 0
    numQ = 0
    # NOTE - (whichSet, dataset(这个就用城市名代替), dbImage(图片路径), utmDb, qImage,
    # utmQ, numDb, numQ, posDistThr,
    # posDistSqThr, nonTrivPosDistSqThr)
    # 我可以用json来保存这些
    # 把后面positive的判定直接改成positive的图片记下来，就不用算距离了，但是需要找到从近到远的距离
    # 把次循环的对应图片分成db和q两个列表，然后把两个列表组成一个list
    # PosImageGT

    # NOTE - 分开两部分train和val，需要给各个城市的每张图设置一个特有的id，将数据记入csv
    
    cities_path = glob(map_database_path)
    header = pd.DataFrame(columns=['place_id', 'city_id', 'map_idx', 'origin_img', 'is_db', 'lon', 'lat', 'lon_LT', 'lat_LT', 'lon_RB', 'lat_RB'])

    for city_path in cities_path:
        city_id = city_path.split('\\')[-1]
        dataset = city_id

        if city_id in VAL_CITIES and val_set_generate == True:
            row_idx = 1
            pass
        else:
            continue

        map_list = glob(os.path.join(city_path, '*.png'))
        map_number = len(map_list)
        for check_idx in map_list:
            check_map = cv2.imread(check_idx)
            if check_map.shape[0] != map_w or check_map.shape[1] != map_h:
                sys.exit('Wrong map shape!')
            else:
                pass
        
        # print(header)
        csv_dataframe = SET_PATH[row_idx][0] + '\\' + city_id + '.csv'
        header.to_csv(csv_dataframe, mode='w', index=False, header=True)
        if map_number < 4:
            completion_number = 4 - map_number
            choose_substitute = random.sample(map_list, completion_number)  # 补齐至少4个地图，要求文件中至少有两个地图
            for substitute in choose_substitute:
                substitute_cv = cv2.imread(substitute)
                substitute_name = substitute.replace('.png','_sub.png')
                cv2.imwrite(substitute_name, substitute_cv)
            map_list = glob(os.path.join(city_path, '*.png'))
            map_number = len(map_list)
        
        reference_number = (map_number + 1) // 2  # 前面的几张原图作为database，让reference的原图数量等于或大于query
        
        lon_lat_path = os.path.join(city_path, 'lon_lat.txt')
        lon_lat_range_np = np.loadtxt(lon_lat_path, delimiter=' ', dtype=np.float64)
        lon_lat_range = lon_lat_range_np.tolist()
        lat_n = lon_lat_range[0][0]
        lat_s = lon_lat_range[0][1]
        lat_res = (lat_s - lat_n) / 5000    # NOTE - 计算每一个像素宽度对应多少纬度变化
        lon_e = lon_lat_range[1][0]
        lon_w = lon_lat_range[1][1]
        lon_res = (lon_e - lon_w) / 5000    # NOTE - 计算每一个像素宽度对应多少经度变化

        # randint_range_x = [[0, map_w - img_w - shift_range]]
        # randint_range_y = [[0, map_h - img_h - shift_range]]

        taken_points = []
        for generate_loop in range(1000):
            # if len(randint_range_x) == 0 or len(randint_range_y) == 0:
            #     print('Not enough random range!')
            #     break
            qPosTmp = []
            dbTmp = []
            dbPoint = []
            qPoint = []
            dbPointPix = []
            qPointPix = []
            place_id = generate_loop
            
            # std_loc_x = random.randint(0, map_w - img_w - shift_range)
            # std_loc_y = random.randint(0, map_h - img_h - shift_range)

            while True:
                std_loc_x = random.randint(0, map_w - img_w - shift_range)
                std_loc_y = random.randint(0, map_h - img_h - shift_range)
                if len(taken_points) == 0:
                    break
                else:
                    neigh = NearestNeighbors(n_neighbors=1, radius=shift_range)
                    neigh.fit(taken_points)
                    idx = neigh.radius_neighbors([[std_loc_x, std_loc_y]], return_distance=False)
                    number = idx.item().size
                if number == 0:
                    break

            # SECTION
            # range_x_idx = random.randint(0, len(randint_range_x) - 1)
            # range_x = randint_range_x[range_x_idx]
            # range_y_idx = random.randint(0, len(randint_range_y) - 1)
            # range_y = randint_range_y[range_y_idx]
            # std_loc_x = random.randint(range_x[0], range_x[1])
            # std_loc_y = random.randint(range_y[0], range_y[1])
            # # range_del_x = [std_loc_x - 2 * shift_range, std_loc_x + 2 * shift_range]
            # # range_del_y = [std_loc_y - 2 * shift_range, std_loc_y + 2 * shift_range]
            # range_del_x = [std_loc_x - shift_range, std_loc_x + shift_range]
            # range_del_y = [std_loc_y - shift_range, std_loc_y + shift_range]
            # # if range_del_x[0] < range_x[0]:
            # #     range_del_x[0] = range_x[0]
            # # if range_del_x[1] > range_x[1]:
            # #     range_del_x[1] = range_x[1]
            # # if range_del_y[0] < range_y[0]:
            # #     range_del_y[0] = range_y[0]
            # # if range_del_y[1] > range_y[1]:
            # #     range_del_y[1] = range_y[1]
            # range_x_new_tmp = [[range_x[0], range_del_x[0]], [range_del_x[1], range_x[1]]]
            # range_y_new_tmp = [[range_y[0], range_del_y[0]], [range_del_y[1], range_y[1]]]
            # range_x_new = []
            # range_y_new = []

            # for i in range(2):
            #     if range_x_new_tmp[i][0] <= range_x_new_tmp[i][1]:
            #         range_x_new.append(range_x_new_tmp[i])
            #     if range_y_new_tmp[i][0] <= range_y_new_tmp[i][1]:
            #         range_y_new.append(range_y_new_tmp[i])

            # randint_range_x.pop(range_x_idx)
            # if len(range_x_new) != 0:
            #     for i in range(len(range_x_new)):
            #         randint_range_x.insert(range_x_idx, range_x_new[-1 - i])
            # randint_range_y.pop(range_y_idx)
            # if len(range_y_new) != 0:
            #     for i in range(len(range_y_new)):
            #         randint_range_y.insert(range_y_idx, range_y_new[-1 - i])

            # del i
            # !SECTION
            random.shuffle(map_list)
            for map_idx in range(map_number):
                if map_idx == 0:
                    loc_x = std_loc_x
                    loc_y = std_loc_y
                else:
                    shift_x = random.randint(0, shift_range)
                    shift_y = random.randint(0, shift_range)
                    loc_x = std_loc_x + shift_x
                    loc_y = std_loc_y + shift_y
                taken_points.append([loc_x, loc_y])
                if map_idx < reference_number:  # 作为database
                    is_db = True
                    row_idx = 0
                else:
                    is_db = False
                    row_idx = 1

                map_origin = cv2.imread(map_list[map_idx])
                map_slice = map_origin[loc_y:loc_y + img_h, loc_x:loc_x + img_w]
                map_resize = cv2.resize(map_slice, (target_w, target_h), interpolation=cv2.INTER_AREA)
                lon_LT = ((loc_x) * lon_res + lon_w)
                lat_LT = ((loc_y) * lat_res + lat_s)
                lon_RB = ((loc_x + img_w) * lon_res + lon_w)
                lat_RB = ((loc_y + img_h) * lat_res + lat_s)
                lon = ((loc_x + img_w / 2) * lon_res + lon_w)
                lat = ((loc_y + img_h / 2) * lat_res + lat_s)

                origin_img = map_list[map_idx].split('\\')[-1].replace('.png', '')
                col_idx = VAL_CITIES.index(city_id)
                img_save_path = DB_Q_VAL_PATH[row_idx][col_idx] + '\\@%s@%s@%s@%s@%s@%s@.png' % (city_id, str(place_id).zfill(4), str(map_idx), origin_img, str(lon), str(lat))

                data_line = pd.DataFrame([[place_id, city_id, map_idx, origin_img, int(is_db), lon, lat, lon_LT, lat_LT, lon_RB, lat_RB]], columns=['place_id', 'city_id', 'map_idx', 'origin_img', 'is_db', 'lon', 'lat', 'lon_LT', 'lat_LT', 'lon_RB', 'lat_RB'])
                data_line.to_csv(csv_dataframe, mode='a', index=False, header=False)

                cv2.imwrite(img_save_path, map_resize)
                VAL_DATASET_ROOT = VAL_ROOT + '\\' + city_id
                img_relative_path = img_save_path.replace(VAL_DATASET_ROOT, '')

                if is_db == 1:
                    dbImage.append(img_relative_path)
                    dbTmp.append(img_relative_path)
                    dbPoint.append((lat, lon))
                    dbPointPix.append((loc_x, loc_y))
                    numDb += 1
                else:
                    qImage.append(img_relative_path)
                    qPosTmp.append(img_relative_path)
                    qPoint.append((lat, lon))
                    qPointPix.append((loc_x, loc_y))
                    numQ += 1
                
            # PosImageGT.append(qPosTmp)
            for db_idx in range(len(dbTmp)):    # 需要排序确定真值里哪个是最近的
                dist_list = []
                dist_pix_list = []
                # NOTE - 这里用的是实际的地理距离确定的，但是其实用左上角的角点的距离确定可能更方便
                for q_idx in range(len(qPosTmp)):
                    # # 用地理距离排序
                    # dist = haversine(dbPoint[db_idx], qPoint[q_idx], unit=Unit.METERS)
                    # dist_list.append(dist)
                    # 用像素距离排序
                    dist_pix = math.dist(dbPointPix[db_idx], qPointPix[q_idx])
                    dist_pix_list.append(dist_pix)
                # dist_info = zip(qPosTmp, dist_list)
                dist_info = zip(qPosTmp, dist_pix_list)
                dist_info = sorted(dist_info, key=lambda x: x[1])
                PosImageGT_line = [dist_info[i][0] for i in range(len(dist_info))]
                gt_idx = [qImage.index(dist_info[i][0]) for i in range(len(dist_info))]
                PosImageGT.append(PosImageGT_line)
                PosImageGT_idx.append(gt_idx)

        json_dict = {'dataset': dataset, 'dataset_root': VAL_ROOT, 'numDb': numDb, 'numQ': numQ, 'dbImage': dbImage, 'qImage': qImage, 'PosImageGT': PosImageGT, 'PosImageGT_idx': PosImageGT_idx}
        
        json_str = json.dumps(json_dict)
        json_save_path = VAL_DATASET_ROOT + '\\info.json'
        with open(json_save_path, 'w') as f:
            json.dump(json_dict, f)

                # tmp = pd.read_csv(csv_dataframe)
                # print(tmp.iloc[-1])

if __name__ == '__main__':
    generate_train_dataset(train_set_generate=True)
    generate_validate_dataset(val_set_generate=True)
    pass