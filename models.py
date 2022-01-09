import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, args, in_ch, out_ch, out_size, norm, bias = False, type = 'up'):
        super().__init__()

        layers = []

        # up or down or same
        if type == 'up':
            if args.generator_upsample:
                layers.append(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False))
                layers.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias = bias))
            else:
                layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias = bias))

            # normalization
            if norm == 'inorm':
                layers.append(nn.InstanceNorm2d(out_ch, affine = True, track_running_stats = True))
            elif norm == 'bnorm':
                layers.append(nn.BatchNorm2d(out_ch))
            elif norm == 'lnorm':
                layers.append(nn.LayerNorm([out_ch, out_size, out_size]))

        elif type == 'down':
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias = bias))

            # normalization
            if norm == 'inorm':
                layers.append(nn.InstanceNorm2d(out_ch, affine = True, track_running_stats = True))
            elif norm == 'bnorm':
                layers.append(nn.BatchNorm2d(out_ch))
            elif norm == 'lnorm':
                layers.append(nn.LayerNorm([out_ch, out_size, out_size]))

        elif type == 'same':
            layers.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias = bias))

            # normalization
            if norm == 'inorm':
                layers.append(nn.InstanceNorm2d(out_ch, affine = True, track_running_stats = True))
            elif norm == 'bnorm':
                layers.append(nn.BatchNorm2d(out_ch))
            elif norm == 'lnorm':
                layers.append(nn.LayerNorm([out_ch, out_size, out_size]))


        if args.nonlinearity == 'leakyrelu':
            layers.append(nn.LeakyReLU(args.slope))
        else:
            layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class Generator(nn.Module):
    def __init__(self, configs, in_ch = 256):
        super(Generator, self).__init__()
        
        self.latent_dim = configs.latent_dim
        self.norm = configs.norm_g
        self.im_size = configs.img_size
        self.in_ch = in_ch

        
        layers = []

        layers.append(ConvBlock(args = configs, in_ch = self.latent_dim, out_ch = in_ch, out_size = 2, norm = self.norm, bias = False, type = 'up'))
        
        # channel up

        if self.im_size == 64:
            ch_list = [256, 512, 1024, 512, 256]
        elif self.im_size == 128:
            ch_list = [256, 512, 1024, 1024, 512, 256]


        for i, out_ch in enumerate(ch_list):
            layers.append(ConvBlock(args = configs, in_ch = in_ch, out_ch = out_ch, out_size = 4 * (2 ** i), norm = self.norm, bias = False, type = 'up'))
            layers.append(ConvBlock(args = configs, in_ch = out_ch, out_ch = out_ch, out_size = 4 * (2 ** i), norm = self.norm, bias = False, type = 'same'))
            in_ch = out_ch


        # To RGB
        layers.append(nn.Conv2d(out_ch, 3, kernel_size=3, stride=1, padding=1, bias = False))
        layers.append(nn.Tanh())
    
        self.main = nn.Sequential(*layers)
    
    def forward(self, z):
        z_tensor = z.view(-1, self.latent_dim, 1, 1)
        out = self.main(z_tensor)
        return out



class Discriminator(nn.Module):
    def __init__(self, configs, out_ch = 64):
        super(Discriminator, self).__init__()

        self.norm = configs.norm_d
        self.im_size = configs.img_size
        
        layers = []
        layers.append(ConvBlock(args = configs, in_ch = 3, out_ch = 64, out_size = self.im_size // 2, norm = self.norm, bias = True, type = 'down'))
        

        in_ch = out_ch
        # channel up
        # 128, 256, 512, 1024, 2048
        # 32, 16, 8, 4, 2, 1. 
        if self.im_size == 64:
            iter_num = 5
        elif self.im_size == 128:
            iter_num = 6

        for i in range(iter_num):
            out_ch = in_ch * 2 if i < 4 else in_ch
            layers.append(ConvBlock(args = configs, in_ch = in_ch, out_ch = out_ch, out_size = self.im_size // (2 ** (i+2)), norm = self.norm, bias = True, type = 'down'))
            in_ch = out_ch
        
        layers.append(nn.Conv2d(in_ch, 1, 3, 1, 1, bias = False))

        if configs.loss == 'bce':
            layers.append(nn.Sigmoid())
        
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        out = self.main(x)
        return out

        
