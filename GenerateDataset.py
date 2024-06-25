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
SET_PATH = []
IMG_PATH = [[], []]

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
        create_noexist_dir(city_img_path)
        IMG_PATH[1].append(city_img_path)

# train_base = BASE_DIR + r'\train'
# val_base = BASE_DIR + r'\val'
# path = []
# train_images_path = train_base + r'\images'
# path.append(train_images_path)
# train_dataframes_path = train_base + r'\dataframes'
# path.append(train_dataframes_path)
# val_images_path = val_base + r'\images'
# path.append(val_images_path)
# val_dataframes_path = val_base + r'\dataframes'
# path.append(val_dataframes_path)
# for i in path:
#     if not os.path.exists(i):
#         os.makedirs(i)



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

def generate_database(train_set_generate = False, val_set_generate = False):
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
        elif val_set_generate == True and city_id in VAL_CITIES:
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
        
        lon_lat_path = os.path.join(city_path, 'lon_lat.txt')
        lon_lat_range_np = np.loadtxt(lon_lat_path, delimiter=' ', dtype=np.float64)
        lon_lat_range = lon_lat_range_np.tolist()
        lat_n = lon_lat_range[0][0]
        lat_s = lon_lat_range[0][1]
        lat_res = (lat_s - lat_n) / 5000    # NOTE - 计算每一个像素宽度对应多少纬度变化
        lon_e = lon_lat_range[1][0]
        lon_w = lon_lat_range[1][1]
        lon_res = (lon_e - lon_w) / 5000    # NOTE - 计算每一个像素宽度对应多少经度变化


        for generate_loop in range(1000):
            place_id = generate_loop
            random.shuffle(map_list)
            std_loc_x = random.randint(0, map_w - img_w - shift_range)
            std_loc_y = random.randint(0, map_h - img_h - shift_range)

            for map_idx in range(map_number):
                if map_idx == 0:
                    loc_x = std_loc_x
                    loc_y = std_loc_y
                else:
                    shift_x = random.randint(0, shift_range)
                    shift_y = random.randint(0, shift_range)
                    loc_x = std_loc_x + shift_x
                    loc_y = std_loc_y + shift_y
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
                else:
                    col_idx = VAL_CITIES.index(city_id)
                cv2.imwrite(IMG_PATH[row_idx][col_idx] +'\\@map%s.png'
                                % ('@' + str(place_id).zfill(4) + '@' + str(map_idx) +'@' + city_id + '@' + str(lon) + '@' + str(lat) + '@'), map_resize)
                # tmp = pd.read_csv(csv_dataframe)
                # print(tmp.iloc[-1])

def generate_val_database():
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

    # NOTE - 分开两部分train和val，需要给各个城市的每个位置设置一个特有的id，将数据记入csv
    
    
    cities_path = glob(map_database_path)

    for city_path in cities_path:
        city_id = city_path.split('\\')[-1]
        if city_id not in VAL_CITIES:
            continue
        map_list = glob(os.path.join(city_path, '*.png'))
        map_number = len(map_list)
        for check_idx in map_list:
            check_map = cv2.imread(check_idx)
            if check_map.shape[0] != map_w or check_map.shape[1] != map_h:
                sys.exit('Wrong map shape!')
            else:
                pass
        header = pd.DataFrame(columns=['place_id', 'city_id', 'lon', 'lat', 'lon_LT', 'lat_LT', 'lon_RB', 'lat_RB'])
        # print(header)
        csv_dataframe = val_dataframes_path + '\\' + city_id + '.csv'
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
        lon_res = (lon_e - lon_w) / 5000    # NOTE - 计算每一个像素宽度对应多少经度变化


        for generate_loop in range(1000):
            place_id = generate_loop
            random.shuffle(map_list)
            std_loc_x = random.randint(0, map_w - img_w - shift_range)
            std_loc_y = random.randint(0, map_h - img_h - shift_range)

            for map_idx in range(map_number):
                if map_idx == 0:
                    loc_x = std_loc_x
                    loc_y = std_loc_y
                else:
                    shift_x = random.randint(0, shift_range)
                    shift_y = random.randint(0, shift_range)
                    loc_x = std_loc_x + shift_x
                    loc_y = std_loc_y + shift_y
                map_origin = cv2.imread(map_list[map_idx])
                map_slice = map_origin[loc_y:loc_y + img_h, loc_x:loc_x + img_w]
                map_resize = cv2.resize(map_slice, (target_w, target_h), interpolation=cv2.INTER_AREA)
                lon_LT = ((loc_x) * lon_res + lon_w)
                lat_LT = ((loc_y) * lat_res + lat_s)
                lon_RB = ((loc_x + img_w) * lon_res + lon_w)
                lat_RB = ((loc_y + img_h) * lat_res + lat_s)
                lon = ((loc_x + img_w / 2) * lon_res + lon_w)
                lat = ((loc_y + img_h / 2) * lat_res + lat_s)
                data_line = pd.DataFrame([[place_id, city_id, lon, lat, lon_LT, lat_LT, lon_RB, lat_RB]], columns=['place_id', 'city_id', 'lon', 'lat', 'lon_LT', 'lat_LT', 'lon_RB', 'lat_RB'])
                data_line.to_csv(csv_dataframe, mode='a', index=False, header=False)
                cv2.imwrite(val_images_path + '\\map%s.png'
                                % ('@' + str(place_id).zfill(4) + '@' + city_id + '@' + str(lon) + '@' + str(lat) + '@'), map_resize)
                # tmp = pd.read_csv(csv_dataframe)
                # print(tmp.iloc[-1])

if __name__ == '__main__':
    generate_database(train_set_generate = True, val_set_generate = True)
    pass