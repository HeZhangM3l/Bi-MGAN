# Overview

A test implementation for the paper "T1-to-T2 MRI images Bidirectional prediction by using Multiple GANs(Bi-MGANs)"

# Environment: 
  python 3.6

# Supported Toolkits
  pytorch
  
  torchvision
  
  numpy
  
  opencv-python
  
# Demo

  1. The pre-trained models latest_net_G_A.pth and latest_net_G_B.pth for T1 predictor and T2 predictor are placed in ./checkpoints/brain_model.

  2. The Brain dataset is placed in ./datasets/brain, which has subfolders testA and testB corresponding T1- and T2-weighted images.
  
  3. Test for bidirectional prediction between T1 and T2 on Brain datasets.
     python test.py --dataroot ./datasets/brain --name brain_model 

  4. The predicted T1 and T2 image will be saved in ./results folder.

# Notes
- The implementation of proposed model is based on cycle-GAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We improve the cycle-GAN by introducing pathological auxiliary information, spectral normalization, localization, pMSE, perceptual loss and edge retention to achieve the bidirectional prediction between T1 and T2 images.
- Due to the samples in SPLP dataset related to private information, if you need the dataset, you can email me (hezhangm3l@gmail.com).
