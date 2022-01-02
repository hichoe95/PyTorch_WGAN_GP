import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os.path
import torch.nn.parallel
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import random
from matplotlib import gridspec
from tqdm import tqdm


from config import *
from misc import *
from models import *
from train import *
from data_loader import *



def main(args):

	train_data = train_loader(configs = args)

	G = Generator(configs = args)
	D = Discriminator(configs = args)

	print(f'Ratio of params ( # of G params / # of D params )  : {get_n_params(G)/get_n_params(D):f}')

	if args.weight_init:
		G.apply(weights_init)
		D.apply(weights_init)
		print('Weights of both G and D are initiated.')

	if args.optim == 'Adam':
		optim_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)
		optim_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
	elif args.optim == 'RMSprop':
		optim_D = optim.RMSprop(D.parameters(), lr=args.lr)
		optim_G = optim.RMSprop(G.parameters(), lr=args.lr)
	else:
		print("You should choose either one, Adam or RMSprop")
		exit(0)

	if args.loss == 'wgangp':
		train(G, D, optim_G, optim_D, train_data, args)
	elif args.loss == 'bce':
		train_bce(G, D, optim_G, optim_D, train_data, args)

	print('Training is finished !!')

if __name__ == '__main__':
	args = parse_args()

	main(args)
