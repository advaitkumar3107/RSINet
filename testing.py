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
from models import *
from metrics import *
from utilities import *
from dataset import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.cuda.empty_cache()
torch.cuda.set_device(0)

device = 'cuda'
torch.cuda.manual_seed(7)
torch.manual_seed(7)
np.random.seed(7)
torch.cuda.empty_cache()

CUDA_LAUNCH_BLOCKING = 1


dataset = dset.ImageFolder('AerialPhoto_split/val/')
dataset = Dataset(dataset, augment = False, training = False)
test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)

def testing(edge_model, inpaint_model, test_loader, postprocess, psnr1):
  edge_model.eval()
  inpaint_model.eval()
#  final_generator.eval()

  psnr = 0.0

  for items in test_loader:
    images, images_gray, edges, masks = items
    images, images_gray, edges, masks = images.cuda(), images_gray.cuda(), edges.cuda(), masks.cuda()

    edges_masked = edges * (1 - masks)
    images_gray_masked = images_gray * (1 - masks) + masks
    inputs = torch.cat((images_gray_masked, edges_masked, masks), dim = 1)
    e_outputs = edge_model(inputs)

    e_outputs = e_outputs * masks + edges * (1 - masks)

    images_masked = (images * (1 - masks).float()) + masks
    inputs = torch.cat((images_masked, e_outputs), dim = 1)
   # inputs = images_masked
    outputs = inpaint_model(inputs)
    outputs_merged = (outputs * masks) + (images * (1 - masks))
   # outputs_merged = torch.cat((images_masked, outputs_merged), dim = 1)

#    outputs_merged = final_generator(outputs_merged)

#    outputs_merged = (outputs_merged * masks) + (images * (1 - masks))

    metric = psnr1(postprocess(images), postprocess(outputs_merged))

    psnr = psnr + metric

  psnr = psnr/len(test_loader)
  print(psnr)


def display(edge_model, inpaint_model, index, dataset, name):
  edge_model.eval()
  inpaint_model.eval()
#  final_generator.eval()

  images, images_gray, edges, masks = dataset.__getitem__(index)
  images, images_gray, edges, masks = images.unsqueeze_(0).cuda(), images_gray.unsqueeze_(0).cuda(), edges.unsqueeze_(0).cuda(), masks.unsqueeze_(0).cuda()

  edges_masked = edges * (1 - masks)
  images_gray_masked = images_gray * (1 - masks) + masks
  inputs = torch.cat((images_gray_masked, edges_masked, masks), dim = 1)
  e_outputs = edge_model(inputs)

  e_outputs = e_outputs * masks + edges * (1 - masks)

  images_masked = (images * (1 - masks).float()) + masks
  inputs = torch.cat((images_masked, e_outputs), dim = 1)
  #inputs = images_masked
  outputs = inpaint_model(inputs)

  outputs_merged = (outputs * masks) + (images * (1 - masks))
#  outputs_merged2 = torch.cat((images_masked, outputs_merged), dim = 1)
#  outputs_merged1 = final_generator(outputs_merged)

#  outputs_merged1 = outputs_merged1 * masks + (images * (1 - masks))

#  outputs = torch.cat((images.detach().cpu(), images_masked.detach().cpu(),outputs.detach().cpu(), outputs_merged1.detach().cpu()), axis = 0)
  outputs = torch.cat((images.detach().cpu(), images_masked.detach().cpu(), outputs.detach().cpu(), outputs_merged.detach().cpu()), axis = 0)

  torchvision.utils.save_image(outputs, str(name) + '.png')



def display_jigsaw(edge_model, inpaint_model, final_generator, index, dataset):
  edge_model.eval()
  inpaint_model.eval()
  final_generator.eval()

  images, images_gray, edges, masks = dataset.__getitem__(index)
  images, images_gray, edges, masks = images.unsqueeze_(0).cuda(), images_gray.unsqueeze_(0).cuda(), edges.unsqueeze_(0).cuda(), masks.unsqueeze_(0).cuda()

#  jumbled_images = jigsaw(images, 16)
#  jumbled_images1 = jigsaw(images, 16)
#  jumbled_images = jumbled_images.cuda()
  
  images_masked = (images * (1 - masks).float()) + masks

#  jumbled_inputs = jumbled_images
#  jumbled_outputs = final_generator(jumbled_inputs)

  outputs = torch.cat((images.detach().cpu(), jumbled_images.detach().cpu(), jumbled_outputs.detach().cpu()), axis = 0)
 # outputs = torch.cat((images.detach().cpu(), images_masked.detach().cpu(), outputs.detach().cpu(), outputs_merged.detach().cpu()), axis = 0)

  torchvision.utils.save_image(outputs, 'grid2.png')




checkpoints = torch.load('aerialphoto_handmade_official_edge_load.ckpt.t7')
edge_generator = checkpoints['edge_generator']

checkpoints = torch.load('aerialphoto_handmade_official_inpaint_load.ckpt.t7')
inpaint_generator = checkpoints['inpaint_generator']

#checkpoints = torch.load('satellite_imagery_masked_our_global_gan_jigsaw_best.ckpt.t7')
#final_generator = checkpoints['generator']

psnr1 = PSNR(255.0)

#testing(edge_generator, inpaint_generator, test_loader, postprocess, psnr1)

for j in range(20):
  display(edge_generator, inpaint_generator, 4*j, dataset, j)
#display_jigsaw(edge_generator, inpaint_generator, final_generator, 117, dataset)