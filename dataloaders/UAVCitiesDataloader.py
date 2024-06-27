import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloaders.UAVCitiesDataset import UAVCitiesDataset
from GenerateDataset import CITIES,TRAIN_CITIES, VAL_CITIES
from . import UAVCitiesValDataset

from prettytable import PrettyTable

# FIXME - 不确定这两个数值是怎么来的，在GSV-cities里就是这么设置的
IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

class UAVCitiesDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 img_per_place=4,
                 min_img_per_place=4,
                 shuffle_all=False,
                #  image_size=(480, 640),
                 image_size=(500, 500),
                 num_workers=4,
                 show_data_stats=True,
                #  cities=TRAIN_CITIES,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 random_sample_from_each_place=True,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = VAL_CITIES
        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

        # NOTE - 训练前的图像变换
        self.train_transform = T.Compose([
            # NOTE - 之前generatedataset的时候已经做了resize，不需要再做了
            # T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),   # NOTE - 对图库进行增广（图像随机旋转、平移、颜色变换）
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        # NOTE - 验证前的图像变换
        self.valid_transform = T.Compose([
            # NOTE - 之前generatedataset的时候已经做了resize，不需要再做了
            # T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])
        
        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all}
        
        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # load train dataloader with reload routine
            self.reload()       # NOTE - reload在后面有定义

            self.val_datasets = []
            for val_set_name in self.val_set_names:
                self.val_datasets.append(UAVCitiesValDataset.get_whole_dataset(dataset_name=val_set_name, input_transform=self.valid_transform))

            if self.show_data_stats:
                self.print_stats()
    def reload(self):
        self.train_dataset = UAVCitiesDataset(
                is_val=False,
                transform=self.train_transform
        )

    def train_dataloader(self):
        self.reload()
        return DataLoader(self.train_dataset, **self.train_loader_config)
    
    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

    def print_stats(self):
        print()
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of cities", f"{len(TRAIN_CITIES)}"])
        table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        table.add_row(["# of images", f'{self.train_dataset.total_nb_images}'])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        print(table.get_string(title="Validation Datasets"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(
            ["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))