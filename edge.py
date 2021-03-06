# -*- coding: utf-8 -*-
"""EdgeConnect_Official

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xkizorZA0Re_PgMXvUz1HKeP4oVFgsF9
"""

# Commented out IPython magic to ensure Python compatibility.
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import pickle
import glob
import random
import math
from random import randint
import scipy
from scipy.misc import imread
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F
import torch.nn.functional as TF
from torchvision import models
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

from dataset import *
from utilities import *
from metrics import *
from models import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.cuda.empty_cache()
torch.cuda.set_device(0)

device = 'cuda'
torch.cuda.manual_seed(7)
torch.manual_seed(7)
np.random.seed(7)
torch.cuda.empty_cache()

CUDA_LAUNCH_BLOCKING = 1

####CBAM Attention BLocks####
train_dataset = dset.ImageFolder('AerialPhoto_split/train/')
val_dataset = dset.ImageFolder('AerialPhoto_split/val/')

train_loader = torch.utils.data.DataLoader(Dataset(train_dataset), batch_size = 4, shuffle = True)
val_loader = torch.utils.data.DataLoader(Dataset(val_dataset, augment = True, training = False), batch_size = 4, shuffle = False)

def train_edge(edge_generator, edge_discriminator, dataloader, epoch, num_epochs, l1_loss, l2_loss, adversarial_loss, edge_gen_optimizer, edge_dis_optimizer, edge_loss):
  
  edge_generator.train()
  edge_discriminator.train()

  for j,items in enumerate(dataloader):
    if j < 15000:
      edge_gen_optimizer.zero_grad()
      edge_dis_optimizer.zero_grad()
      
      e_gen_loss = 0
      e_dis_loss = 0

      images, images_gray, edges, masks = items
      images, images_gray, edges, masks = images.cuda(), images_gray.cuda(), edges.cuda(), masks.cuda()

      edges_masked = edges * (1 - masks)
      images_gray_masked = images_gray * (1 - masks) + masks
      inputs = torch.cat((images_gray_masked, edges_masked, masks), dim = 1)
      e_outputs = edge_generator(inputs)

      ####  edge discriminator loss   ####
      e_dis_input_real = torch.cat((images_gray, edges), dim = 1)
      e_dis_input_fake = torch.cat((images_gray, e_outputs.detach()), dim = 1)
      e_dis_real, e_dis_real_feat = edge_discriminator(e_dis_input_real)
      e_dis_fake, e_dis_fake_feat = edge_discriminator(e_dis_input_fake)
      
      target_real_label = 1.0
      target_fake_label = 0.0
      real_label = torch.tensor(target_real_label)
      fake_label = torch.tensor(target_fake_label)

      e_dis_labels = real_label.expand_as(e_dis_real)
      e_dis_real_loss = adversarial_loss(e_dis_real, e_dis_labels.cuda())
      
      e_labels = fake_label.expand_as(e_dis_fake)
      e_dis_fake_loss = adversarial_loss(e_dis_fake, e_labels.cuda())

      e_dis_loss += (e_dis_real_loss + e_dis_fake_loss) / 2


      ### edge generator loss #####
      e_gen_input_fake = torch.cat((images_gray, e_outputs), dim = 1)
      e_gen_fake, e_gen_fake_feat = edge_discriminator(e_gen_input_fake)
      
      e_gen_labels = real_label.expand_as(e_gen_fake)
      e_gen_gan_loss = adversarial_loss(e_gen_fake, e_gen_labels.cuda())
      e_gen_loss += e_gen_gan_loss




      ### edge generator feature matching loss ###
      e_gen_fm_loss = 0

      for i in range(len(e_dis_real_feat)):
        e_gen_fm_loss += l1_loss(e_gen_fake_feat[i], e_dis_real_feat[i].detach())
      e_gen_fm_loss = e_gen_fm_loss * 10
      e_gen_loss += e_gen_fm_loss

      edge_loss.append(e_gen_loss)

      e_dis_loss.backward()
      edge_dis_optimizer.step()

      e_gen_loss.backward()
      edge_gen_optimizer.step()

      print('Epoch : %d/%d \t  Iters : %d/99  \t Edge Generator Loss : %.4f \t Edge discriminator loss : %.4f' % (epoch + 1, num_epochs, j , e_gen_loss, e_dis_loss))

    else:
      break


def validation_edge(edge_generator, edge_discriminator, val_loader, l1_loss, adversarial_loss, epoch, edge_val_loss):
  edge_generator.eval()
  edge_discriminator.eval()

  global edge_best_loss
  edge_val_loss.append(0)

  for j,items in enumerate(val_loader):
    e_gen_loss = 0
    e_dis_loss = 0

    images, images_gray, edges, masks = items
    images, images_gray, edges, masks = images.cuda(), images_gray.cuda(), edges.cuda(), masks.cuda()

    edges_masked = edges * (1 - masks)
    images_gray_masked = images_gray * (1 - masks) + masks
    inputs = torch.cat((images_gray_masked, edges_masked, masks), dim = 1)
    e_outputs = edge_generator(inputs)


    ####  edge discriminator loss   ####
    e_dis_input_real = torch.cat((images_gray, edges), dim = 1)
    e_dis_input_fake = torch.cat((images_gray, e_outputs.detach()), dim = 1)
    e_dis_real, e_dis_real_feat = edge_discriminator(e_dis_input_real)
    e_dis_fake, e_dis_fake_feat = edge_discriminator(e_dis_input_fake)
    
    target_real_label = 1.0
    target_fake_label = 0.0
    real_label = torch.tensor(target_real_label)
    fake_label = torch.tensor(target_fake_label)

    e_dis_labels = real_label.expand_as(e_dis_real)
    e_dis_real_loss = adversarial_loss(e_dis_real, e_dis_labels.cuda())
    
    e_labels = fake_label.expand_as(e_dis_fake)
    e_dis_fake_loss = adversarial_loss(e_dis_fake, e_labels.cuda())

    e_dis_loss += (e_dis_real_loss + e_dis_fake_loss) / 2


    ### edge generator loss #####
    e_gen_input_fake = torch.cat((images_gray, e_outputs), dim = 1)
    e_gen_fake, e_gen_fake_feat = edge_discriminator(e_gen_input_fake)
    
    e_gen_labels = real_label.expand_as(e_gen_fake)
    e_gen_gan_loss = adversarial_loss(e_gen_fake, e_gen_labels.cuda())
    e_gen_loss += e_gen_gan_loss

    ### edge generator feature matching loss ###
    e_gen_fm_loss = 0

    for i in range(len(e_dis_real_feat)):
      e_gen_fm_loss += l1_loss(e_gen_fake_feat[i], e_dis_real_feat[i].detach())
    e_gen_fm_loss = e_gen_fm_loss * 10
    e_gen_loss += e_gen_fm_loss

    edge_val_loss[-1] = edge_val_loss[-1] + e_gen_loss.data
    print('Val_loss = %.4f' % (edge_val_loss[-1]/(j+1)))

  edge_val_loss[-1] = edge_val_loss[-1]/len(val_loader)

  if edge_best_loss > edge_val_loss[-1]:
    edge_best_loss = edge_val_loss[-1]
    print('Saving...')

    state = {'edge_generator' : edge_generator, 'edge_discriminator' : edge_discriminator}
    torch.save(state, 'aerialphoto_rectangle_official_edge_best.ckpt.t7')
    
    f = open("edge_best_loss.txt", "w")
    f.write(str(edge_best_loss.item()))
    f.close()    

edge_loss = []
edge_val_loss = []

torch.cuda.empty_cache()

checkpoints = torch.load('aerialphoto_rectangle_official_edge_load.ckpt.t7')
edge_generator = checkpoints['edge_generator']
edge_discriminator = checkpoints['edge_discriminator']

#edge_generator = EdgeGenerator().cuda()
#edge_discriminator = Discriminator(2).cuda()

edge_gen_opt = optim.Adam(params = edge_generator.parameters(), lr = float(0.0001), betas = (0.0, 0.9))
edge_dis_opt = optim.Adam(params = edge_discriminator.parameters(), lr = float(0.0001), betas = (0.0, 0.9))

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
adversarial_loss = nn.BCELoss()

num_epochs = 3

h = open('edge_best_loss.txt','r')
edge_best_loss = float(h.read())
h.close()

#edge_best_loss = 1e5

for epoch in range(num_epochs):	
  train_edge(edge_generator, edge_discriminator, train_loader, epoch, num_epochs, l1_loss, l2_loss, adversarial_loss, edge_gen_opt, edge_dis_opt, edge_loss)
  validation_edge(edge_generator, edge_discriminator, val_loader, l1_loss, adversarial_loss, epoch, edge_val_loss)    
  checkpoints = {'edge_generator' : edge_generator, 'edge_discriminator' : edge_discriminator}
  torch.save(checkpoints, 'aerialphoto_rectangle_official_edge_load.ckpt.t7')
