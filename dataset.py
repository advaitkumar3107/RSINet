from utilities import *
import random
import os
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

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset, augment = True, training = True, transform = False):
    super(Dataset, self).__init__()
    self.augment = augment
    self.training= training
    self.dataset = dataset
    self.transform = transform

    self.input_size = 256
    self.sigma = 2.0
    self.edge = 1
    self.mask = 3
    self.nms = 1

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    image, mask = self.dataset.__getitem__(index)
    img = np.asarray(image)

    size = self.input_size

    if size != 0:
      img = self.resize(img, size, size)

    img_gray = rgb2gray(img)
    
    mask = self.load_mask(img, index)

    # load edge
    edge = self.load_edge(img_gray, mask)

    # augment data
    if self.augment and np.random.binomial(1, 0.5) > 0:
      img = img[:, ::-1, ...]
      img_gray = img_gray[:, ::-1, ...]
      edge = edge[:, ::-1, ...]
      mask = mask[:, ::-1, ...]
    
    if self.transform:
      transformation = transforms.Compose([transforms.RandomAffine(180, scale = (0.5, 1.5)), transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1)])
      img1 = Image.fromarray(img)      
      transformed_img = transformation(img1)
      transformed_img = np.asarray(transformed_img)

      if size != 0:
        transformed_img = self.resize(transformed_img, size, size)

      transformed_img_gray = rgb2gray(transformed_img)
      transformed_img_edge = self.load_edge(transformed_img_gray, mask)
  
      return self.to_tensor(img), self.to_tensor(transformed_img), self.to_tensor(transformed_img_gray), self.to_tensor(transformed_img_edge), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)
  
    else:
      return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)
    

  def load_edge(self, img, mask):
      sigma = self.sigma

      # in test mode images are masked (with masked regions),
      # using 'mask' parameter prevents canny to detect edges for the masked regions
      mask = None if self.training else (1 - mask / 255).astype(np.bool)

      # canny
      if self.edge == 1:
          # no edge
          if sigma == -1:
              return np.zeros(img.shape).astype(np.float)

          # random sigma
          if sigma == 0:
              sigma = random.randint(1, 4)

          return canny(img, sigma=sigma, mask=mask).astype(np.float)

  def load_mask(self, img, index):
      imgh, imgw = img.shape[0:2]
      mask_type = self.mask

      height = random.randint(64,145)
      width = random.randint(64,145)
      percent = random.randint(10,90)

      # random block
      if mask_type == 1:
          return create_mask(imgw, imgh, height, width)

      if mask_type == 2:
          return salt_and_pepper(percent, imgw, imgh)

      if mask_type == 3:
          ids = os.listdir('irregular_masks')
          random_id = random.choice(ids)
          image_path = 'irregular_masks/' + random_id
          return irregular_masks(image_path)

   
  def to_tensor(self, img):
      img = Image.fromarray(img)
      img_t = F.to_tensor(img).float()
      return img_t

  def resize(self, img, height, width, centerCrop=True):
      imgh, imgw = img.shape[0:2]

      if centerCrop and imgh != imgw:
          # center crop
          side = np.minimum(imgh, imgw)
          j = (imgh - side) // 2
          i = (imgw - side) // 2
          img = img[j:j + side, i:i + side, ...]

      img = scipy.misc.imresize(img, [height, width])

      return img



  def create_iterator(self, batch_size):
      while True:
          sample_loader = DataLoader(
              dataset=self,
              batch_size=batch_size,
              drop_last=True
          )

          for item in sample_loader:
              yield item