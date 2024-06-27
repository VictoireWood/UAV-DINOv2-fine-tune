import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from GenerateDataset import CITIES,TRAIN_CITIES, VAL_CITIES

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),     # NOTE - mean和std是沿用了GSVCities的代码。
])

# NOTE: Hard coded path to dataset folder 
# BASE_PATH = '../data/GSVCities/'
BASE_PATH = r'E:\GeoVINS\AerialVL\vpr_training_data'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

# TRAIN_CITIES = [
#     'CITY_1',
#     'CITY_2',
#     'CITY_3',
#     'CITY_4',
#     'CITY_5',
#     'CITY_6',
#     'CITY_7'
# ]

class UAVCitiesDataset(Dataset):
    def __init__(self,
                #  cities=TRAIN_CITIES,
                #  is_val=False,
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH
                 ):
        super(UAVCitiesDataset, self).__init__()
        self.base_path = base_path + r'\train'
        self.cities = TRAIN_CITIES
        # if is_val == False:
        #     self.base_path = base_path + r'\train'
        #     self.cities = TRAIN_CITIES
        # else:
        #     self.base_path = base_path + r'\val'
        #     self.cities = VAL_CITIES

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place  # 目前只能选True，因为没有其他可以排序的参数
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # get all unique place ids
        # NOTE - 等价self.places_ids = pd.unique(self.dataframe['place_id'])
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[0]}.csv')  # NOTE - 读取数据csv文件
        df = df.sample(frac=1)  # shuffle the city dataframe

        # append other cities one by one
        for i in range(1, len(self.cities)):
            # SECTION - 原始
            tmp_df = pd.read_csv(
                self.base_path+'Dataframes/'+f'{self.cities[i]}.csv')
            # !SECTION
            

            prefix = i
            tmp_df = tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**4)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe

            df = pd.concat([df, tmp_df], ignore_index=True)
            # keep only places depicted by at least min_img_per_place images
            res = df[df.groupby('place_id')['place_id'].transform(
                'size') >= self.min_img_per_place]
            return res.set_index('place_id')
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        # NOTE - place至少对应四行
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        # SECTION - 原始有年份时间信息
        # else:  # always get the same most recent images
        #     place = place.sort_values(
        #         by=['year', 'month', 'lat'], ascending=False)
        #     place = place[: self.img_per_place]
        # !SECTION

        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tensor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)
    
    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)
    
    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))
    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']
        
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**4  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(4)

        lat, lon = str(row['lat']), str(row['lon'])
        name = '@map@'+pl_id+'@'+city+'@'+lon+'@'+lat+'@'+lon+'@.png'
        return name
