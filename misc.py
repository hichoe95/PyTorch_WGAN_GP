import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


def weights_init(model):

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def minmax(x):
    return (x - x.min())/(x.max() - x.min())

def gradient_penalty(y, x, device):

    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs = y,
                          inputs = x,
                          grad_outputs = weight,
                          retain_graph = True,
                          create_graph = True,
                          only_inputs = True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim = 1))

    return torch.mean((dydx_l2norm-1)**2)



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        
        pp += nn
    return pp
