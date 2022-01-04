import argparse
import os




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--main_gpu", type=int, default=4, help="main gpu index")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Tensorboard")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint', help="full name is './checkponit'.format(main_gpu) ")
    parser.add_argument("--log_dir", type=str, default = 'runs', help="dir for tensorboard")
    parser.add_argument("--image_name", type = str, default = 'gen_images', help="sample image name")
    parser.add_argument("--train_data", type = str, default = 'celeba', help="celeba or ffhq")

    # optimizer
    parser.add_argument("--optim", type=str, default='Adam', help="Adam or RMSprop")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--beta1", type=float, default = 0.5, help="For Adam optimizer.")
    parser.add_argument("--beta2", type=float, default = 0.999, help="For Adam optimizer.")


    # model
    parser.add_argument("--latent_dim", type = int, default=128, help="dimension of latent vector")
    parser.add_argument("--generator_upsample", type=bool, default=False, help="if False, using ConvTranspose.")
    parser.add_argument("--weight_init", type=bool, default=False, help="weight init from normal dist")
    parser.add_argument("--normalization", type=str, default='None', help="inorm : instancenorm, bnorm : batchnorm, or None")
    parser.add_argument("--nonlinearity", type=str, default='relu', help="relu or leakyrelu")
    parser.add_argument("--slope", type=float, default = 0.2, help="if using leakyrelu, you can use this option.")

    # training
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")    
    parser.add_argument("--iter_num", type=int, default=200000, help="number of iterations of training")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--loss", type=str, default='wgangp', help="wgangp or bce, default is wgangp")
    parser.add_argument("--n_critic", type=int, default=3, help="number of training steps for discriminator per iter")
    parser.add_argument("--lambda_gp", type=float, default=10, help="amount of gradient penalty loss")
    
    config = parser.parse_args()
    
    return config
