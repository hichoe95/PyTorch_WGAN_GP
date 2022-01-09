# PyTorch WGAN GP

This repository is only for training.
Later, I will provide/upload pretrained weight.  

![](https://github.com/hichoe95/WGAN_GP_CelebAHQ/blob/main/image/gpu_4.png?raw=true)


## Version

- pytorch=1.4.0
- pyyhon3.6
- cuda10.0.x
- cudnn7.6.3

* environment.yaml should be used for reference only, since it has too many dependencies.



## Dataset
|Dataset| [CelebA HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ) | [FFHQ(thumbnails)](https://drive.google.com/drive/folders/1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv) |
|:---:|:---:|:---:|
|size|1024 x 1024| 128 x 128|
|# of images | 30000 | 70000 |

(In training, **CelebA HQ** images are resized in 128 x 128 or 64 x 64)

You should change the name of directory root in data_loder.py.

## Options and Help

```bash
wgan_gp$ python main.py -h
usage: main.py [-h] [--main_gpu MAIN_GPU] [--use_tensorboard USE_TENSORBOARD]
               [--checkpoint_dir CHECKPOINT_DIR] [--log_dir LOG_DIR]
               [--image_name IMAGE_NAME] [--train_data TRAIN_DATA]
               [--optim OPTIM] [--lr LR] [--betas BETAS]
               [--latent_dim LATENT_DIM]
               [--generator_upsample GENERATOR_UPSAMPLE]
               [--weight_init WEIGHT_INIT] [--normalization NORMALIZATION]
               [--nonlinearity NONLINEARITY] [--slope SLOPE]
               [--batch_size BATCH_SIZE] [--iter_num ITER_NUM]
               [--img_size IMG_SIZE] [--loss LOSS] [--n_critic N_CRITIC]
               [--lambda_gp LAMBDA_GP]

optional arguments:
  -h, --help            show this help message and exit
  --main_gpu MAIN_GPU   main gpu index
  --use_tensorboard USE_TENSORBOARD
                        Tensorboard
  --checkpoint_dir CHECKPOINT_DIR
                        full name is './checkponit'.format(main_gpu)
  --log_dir LOG_DIR     dir for tensorboard
  --image_name IMAGE_NAME
                        sample image name
  --train_data TRAIN_DATA
                        celeba or ffhq
  --optim OPTIM         Adam or RMSprop
  --lr LR               learning rate
  --betas BETAS         For Adam optimizer.
  --latent_dim LATENT_DIM
                        dimension of latent vector
  --generator_upsample GENERATOR_UPSAMPLE
                        if False, using ConvTranspose.
  --weight_init WEIGHT_INIT
                        weight init from normal dist
  --normalization NORMALIZATION
                        inorm : instancenorm, bnorm : batchnorm, or None
  --nonlinearity NONLINEARITY
                        relu or leakyrelu
  --slope SLOPE         if using leakyrelu, you can use this option.
  --batch_size BATCH_SIZE
                        size of the batches
  --iter_num ITER_NUM   number of iterations of training
  --img_size IMG_SIZE   size of each image dimension
  --loss LOSS           wgangp or bce, default is wgangp
  --n_critic N_CRITIC   number of training steps for discriminator per iter
  --lambda_gp LAMBDA_GP
                        amount of gradient penalty loss

```

## Training

When I use 'RMSprop', it has the best training performance.

```bash
wgan_gp$ python main.py --main_gpu 4 \
                        --log_dir gpu4 \
                        --train_data celeba
                        --latent_dim 128 \
                        --image_name gpu_4.png \
                        --batch_size 32 \
                        --n_critic 5 \
                        --lr 0.00005 \
                        --lambda_gp 10 \
                        --optim RMSprop \
                        --generator_upsample True \
                        --norm_g bnorm \
                        --norm_d lnorm \
                        --nonlinearity leakyrelu \
                        --slope 0.2 \
                        --loss wgangp
```
### Results - CelebA HQ
![](https://github.com/hichoe95/WGAN_GP_CelebAHQ/blob/main/image/gpu_4.png?raw=true)

## Tensorboard

To open tensorbaord window in local, if you run it on remote server, you should follow this command in local.

```console
ssh -NfL localhost:8898:localhost:6009 [USERID]@[IP]
```

'8898' is arbitrary port number for local , and '6009' is arbitrary port number for remote.

