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

## mask creation function ##
def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask

def salt_and_pepper(percent, height, width):
    num_pixels = percent*height*width/100
    num_pixels = int(math.sqrt(num_pixels))
    mask = np.zeros((height,width))

    for i in range(num_pixels):
        for j in range(num_pixels):
            mask[i][j] = 1.0

    mask = mask.ravel()
    np.random.shuffle(mask)
    mask = mask.reshape((height,width))

    return mask



def compute_gram(x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)

    return G


def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


def jigsaw(image, num_patches):
    jumbled_data = image.clone()
    sqrt_num_patches = int(math.sqrt(num_patches))
    assert (math.sqrt(num_patches) **2 - num_patches)==0
    perm = np.random.permutation(np.arange(num_patches))
    s = image.shape[2]
    patch_size = math.floor(s/sqrt_num_patches)

    for x in range(sqrt_num_patches):
        for y in range(sqrt_num_patches):
            l , m = math.floor(perm[sqrt_num_patches*x+y]/sqrt_num_patches), perm[sqrt_num_patches*x+y]%sqrt_num_patches
            jumbled_data[:,:,math.floor(x*patch_size):math.ceil((x+1)*patch_size),math.floor(y*patch_size):math.ceil((y+1)*patch_size)] = image[:,:,math.floor(l*patch_size):math.ceil((l+1)*patch_size),math.floor(m*patch_size):math.ceil((m+1)*patch_size)]
    return jumbled_data

def cosine_similarity(image1, image2):
    numerator = (torch.sum(image1 * image2)) ** 2
    denominator = torch.sum(image1 ** 2) * torch.sum(image2 ** 2)
    return numerator/denominator

def irregular_masks(path):
    img = Image.open(path)
    img = img.resize((256,256))
    img = np.asarray(img)
    img = (img == 0.0).astype(float)
    
    return img
    