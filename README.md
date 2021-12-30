# wgan_gp_clelbA_HQ

This repository is only for training.

## Version

pytorch=1.4.0=py3.6_cuda10.0.130_cudnn7.6.3_0

* environment.yaml should be used for reference only, since it has too many dependencies.

## dataset - CelebA HQ
size : 1024 x 1024.
In training, these images are resized in 128 x 128.

### download link
* https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ


## options and help

```bat
python main.py -h
usage: main.py [-h] [--main_gpu MAIN_GPU] [--use_tensorboard USE_TENSORBOARD]
               [--log_dir LOG_DIR] [--image_name IMAGE_NAME]
               [--iter_num ITER_NUM] [--img_size IMG_SIZE]
               [--latent_dim LATENT_DIM] [--batch_size BATCH_SIZE]
               [--n_critic N_CRITIC] [--lr LR] [--lambda_gp LAMBDA_GP]
               [--optim OPTIM] [--generator_upsample GENERATOR_UPSAMPLE]
               [--normalization NORMALIZATION] [--nonlinearity NONLINEARITY]
               [--slope SLOPE]

optional arguments:
  -h, --help            show this help message and exit
  --main_gpu MAIN_GPU   main gpu index for training
  --use_tensorboard USE_TENSORBOARD
                        Tensorboard
  --log_dir LOG_DIR     dir for tensorboard
  --image_name IMAGE_NAME
  --iter_num ITER_NUM   number of iterations of training
  --img_size IMG_SIZE   size of each image dimension
  --latent_dim LATENT_DIM
                        dimension of latent vector
  --batch_size BATCH_SIZE
                        size of the batches
  --n_critic N_CRITIC   number of training steps for discriminator per iter
  --lr LR               learning rate
  --lambda_gp LAMBDA_GP
                        amount of gradient penalty loss
  --optim OPTIM         choose only Adam or RMSprop
  --generator_upsample GENERATOR_UPSAMPLE
                        if False, using ConvTranspose.
  --normalization NORMALIZATION
                        inorm : instancenorm, bnorm : batchnorm, or None
  --nonlinearity NONLINEARITY
                        relu or leakyrelu
  --slope SLOPE         if using leakyrelu, you can use this option

```

## Usage example

```bat
python main.py --main_gpu 1 \
                --use_tensorboard True \
                --log_dir gpu1 \
                --latent_dim 128 \
                --image_name gpu_1.png \
                --batch_size 16 \
                --n_critic 5 \
                --lr 0.0002 \
                --lambda_gp 10 \
                --optim Adam \
                --generator_upsample True \
                --normalization inorm \
                --nonlinearity leakyrelu \
                --slope 0.1
```

