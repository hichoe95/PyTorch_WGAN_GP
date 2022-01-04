import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os.path
import torch.utils.data as data
import torchvision.datasets as datasets
import random
from PIL import Image
import PIL



class Data_Loader(data.Dataset):
    def __init__(self, train_data, transform, label, mode):
        self.train_data = train_data
        self.transform = transform
        self.label = label
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.file_list = []

        self.load_train_data()
        self.preprocess()
        
        if mode == 'train':
            self.num_imgs = len(self.train_dataset)
        else:
            self.num_imgs = len(self.test_dataset)
    
    def load_train_data(self,):
        
        if self.train_data == 'celeba':
            self.train_data_dir = '/data/hwanil/CelebA_HQ/data1024x1024/'
            self.file_list.extend(os.listdir(self.train_data_dir))

        elif self.train_data == 'ffhq':
            self.train_data_dir = '/data/hwanil/ffhq-dataset/thumbnails128x128/'

            folders = os.listdir(self.train_data_dir)
            for f in folders:
                folder_name = os.path.join(self.train_data_dir, f)
                if os.path.isdir(folder_name):
                    self.file_list.extend([os.path.join(f, img) for img in os.listdir(folder_name)])

        print("Total {:d} {} images.".format(len(self.file_list), self.train_data))

    def preprocess(self):

        random.seed(1234)
        
        for i, file in enumerate(self.file_list):
            self.train_dataset.append([file, self.label])
            
        print('Finished preprocessing...')
        
    def __getitem__(self,index):
        dataset = self.train_dataset # if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        
        image = Image.open(os.path.join(self.train_data_dir, filename))
        
        return self.transform(image), label
        
    def __len__(self):
        return self.num_imgs


def get_loader(configs, label, mode='train', num_workers=1):
    
    transform = transforms.Compose([transforms.Resize(configs.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = Data_Loader(train_data = configs.train_data, transform = transform , label = label, mode = mode)
    
    data_loader = data.DataLoader(dataset = dataset,
                                  batch_size = configs.batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader



def train_loader(configs):
    
    batch_size = configs.batch_size
    train_data = get_loader(configs,
                            label = 1,
                            mode = 'train',
                            num_workers = 1)
    return train_data

