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
    def __init__(self, img_dir, transform, label, mode):
        self.img_dir = img_dir
        self.transform = transform
        self.label = label
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()
        
        if mode == 'train':
            self.num_imgs = len(self.train_dataset)
        else:
            self.num_imgs = len(self.test_dataset)
    
    def preprocess(self):
        file_list = os.listdir(self.img_dir)
        
        print("Total {:d} images.".format(len(file_list)))

        random.seed(1234)
        
        for i, file in enumerate(file_list):
            
            # if (i+1)< 500:
            #     self.test_dataset.append([random.choice(file_list), self.label])
            # else:
                # self.train_dataset.append([random.choice(file_list), self.label])
            self.train_dataset.append([file, self.label])
            
        print('Finished preprocessing...')
        
    def __getitem__(self,index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        
        image = Image.open(os.path.join(self.img_dir, filename))
        
        return self.transform(image), label
        
    def __len__(self):
        return self.num_imgs


def get_loader(configs, img_dir, label, mode='train', num_workers=1):
    
    transform = transforms.Compose([transforms.Resize(configs.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = Data_Loader(img_dir = img_dir , transform = transform , label = label, mode = mode)
    
    data_loader = data.DataLoader(dataset = dataset,
                                  batch_size = configs.batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader



def train_loader(configs, root = '../../../../data/hwanil/CelebA_HQ/data1024x1024/'):
    
    batch_size = configs.batch_size
    train_data = get_loader(configs,
                            root,
                            label = 1,
                            mode = 'train',
                            num_workers = 1)
    return train_data

