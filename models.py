import random
import math
from random import randint
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

from init_weights import *
from attention import *


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.c64 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.d128 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))

        self.d256 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True))

        blocks = []
        for _ in range(4):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        blocks1 = []
        for _ in range(4):
            block1 = ResnetBlock(256, 2)
            blocks1.append(block1)

        self.middle1 = nn.Sequential(*blocks)
        self.middle2 = nn.Sequential(*blocks1)

        self.u128 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))

        self.u64 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.out = nn.Sequential(nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0))

  #      self.attention1 = GridAttentionBlock2D(256,256)
  #      self.attention2 = GridAttentionBlock2D(128,256)
  #      self.attention3 = GridAttentionBlock2D(64,256)

  #      self.skip1 = nn.Sequential(nn.Conv2d(64,128,1,2,padding = 0), nn.BatchNorm2d(128))
  #      self.skip2 = nn.Sequential(nn.Conv2d(128,256,1,2,padding = 0), nn.BatchNorm2d(256))

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.c64(x)
  #      skip = self.skip1(x)
  #      skip = self.skip2(skip)

        x = self.d128(x)
        x = self.d256(x)
  #      x = x + skip
  #      gate1 = x

        x = self.middle1(x)
  #      gate2 = x
        
        x = self.middle2(x)
  #      gate3 = x
        
  #      x = x + self.attention1(x, gate1)
        x = self.u128(x)
  #      x = x + self.attention2(x, gate2)
        x = self.u64(x)
  #      x = x + self.attention3(x, gate3)
        x = self.out(x)

        x = (torch.tanh(x) + 1) / 2

        return x


class EdgeGenerator(BaseNetwork):
    def __init__(self, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.c64 = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))


        self.d128 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))

        self.d256 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True))
        
  #      self.attention1 = GridAttentionBlock2D(256,256)
  #      self.attention2 = GridAttentionBlock2D(128,256)
  #      self.attention3 = GridAttentionBlock2D(64,256)

  #      self.skip1 = nn.Sequential(nn.Conv2d(64,128,1,2,padding = 0), nn.BatchNorm2d(128))
  #      self.skip2 = nn.Sequential(nn.Conv2d(128,256,1,2,padding = 0), nn.BatchNorm2d(256))

        blocks = []
        for _ in range(4):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle1 = nn.Sequential(*blocks)

        blocks1 = []
        for _ in range(4):
          block1 = ResnetBlock(256,2, use_spectral_norm = use_spectral_norm)
          blocks1.append(block1)

        self.middle2 = nn.Sequential(*blocks1)

        self.u128 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))

        self.u64 = nn.Sequential(spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.out =  nn.Sequential(nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0))

        if init_weights:
            self.init_weights()

    def forward(self, x):

        x = self.c64(x)
  #      skip = self.skip1(x)
  #      skip = self.skip2(skip)

        x = self.d128(x)
        x = self.d256(x)
  #      x = x + skip
  #      gate1 = x

        x = self.middle1(x)
  #      gate2 = x

        x = self.middle2(x)
  #      gate3 = x

  #      x = x + self.attention1(x, gate1)
        x = self.u128(x)

  #      x = x + self.attention2(x, gate2)
        x = self.u64(x)

  #      x = x + self.attention3(x, gate3)
        x = self.out(x)

        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class final_generator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(final_generator, self).__init__()

        self.c64 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.d128 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))

        self.d256 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True))

        blocks = []
        for _ in range(4):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        blocks1 = []
        for _ in range(4):
            block1 = ResnetBlock(256, 2)
            blocks1.append(block1)

        self.middle1 = nn.Sequential(*blocks)
        self.middle2 = nn.Sequential(*blocks1)

        self.u128 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))

        self.u64 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.out = nn.Sequential(nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0))


        self.attention1 = GridAttentionBlock2D(256,256)
        self.attention2 = GridAttentionBlock2D(128,256)
        self.attention3 = GridAttentionBlock2D(64,256)

        self.skip1 = nn.Sequential(nn.Conv2d(64,128,1,2,padding = 0), nn.BatchNorm2d(128))
        self.skip2 = nn.Sequential(nn.Conv2d(128,256,1,2,padding = 0), nn.BatchNorm2d(256))

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.c64(x)
        skip = self.skip1(x)
        skip = self.skip2(skip)

        x = self.d128(x)
        x = self.d256(x)
        x = x + skip
        gate1 = x

        x = self.middle1(x)
        gate2 = x
        
        x = self.middle2(x)
        gate3 = x
        
        x = x + self.attention1(x, gate1)
        x = self.u128(x)
        x = x + self.attention2(x, gate2)
        x = self.u64(x)
        x = x + self.attention3(x, gate3)
        x = self.out(x)

        x = (torch.tanh(x) + 1) / 2

        return x



class final_discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(final_discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

      
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs





class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
