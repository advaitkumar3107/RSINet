from torch import autograd
from torch import optim
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
from dataset import dataset_prepare
import matplotlib.pyplot as plt
import cv2
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.cuda.empty_cache()
torch.cuda.set_device(0)

#torch.backends.cudnn.enabled = True

device = 'cuda'
torch.cuda.manual_seed(7)
torch.manual_seed(7)
np.random.seed(7)
torch.cuda.empty_cache()

CUDA_LAUNCH_BLOCKING = 1


### Weights initialization ###
def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2,
                 sub_sample_factor=(2,2)):
        super(_GridAttentionBlockND, self).__init__()

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
          self.inter_channels = in_channels // 2
          if self.inter_channels == 0:
              self.inter_channels = 1

          conv_nd = nn.Conv2d
          bn = nn.BatchNorm2d
          self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

        # Define the operation
        self.operation_function = self._concatenation



    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''
        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = torch.nn.functional.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = torch.nn.functional.relu(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.nn.functional.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = torch.nn.functional.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None,
                 sub_sample_factor=(2,2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )


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
        
        self.attention1 = GridAttentionBlock2D(256,256)
        self.attention2 = GridAttentionBlock2D(128,256)
        self.attention3 = GridAttentionBlock2D(64,256)

        self.skip1 = nn.Sequential(nn.Conv2d(64,128,1,2,padding = 0), nn.BatchNorm2d(128))
        self.skip2 = nn.Sequential(nn.Conv2d(128,256,1,2,padding = 0), nn.BatchNorm2d(256))

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


class final_generator(BaseNetwork):
    def __init__(self, channels = 3, residual_blocks=8, init_weights=True):
        super(final_generator, self).__init__()

        self.c64 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels= channels, out_channels=64, kernel_size=7, padding=0),
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



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


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


def postprocess(self, img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()



class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset, augment = True, training = True):
    super(Dataset, self).__init__()
    self.augment = augment
    self.training= training
    self.dataset = dataset

    self.input_size = 256
    self.sigma = 2.0
    self.edge = 1
    self.mask = 2
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
    edge = self.load_edge(img_gray, index, mask)

    # augment data
    if self.augment and np.random.binomial(1, 0.5) > 0:
      img = img[:, ::-1, ...]
      img_gray = img_gray[:, ::-1, ...]
      edge = edge[:, ::-1, ...]
      mask = mask[:, ::-1, ...]

    

    return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)


  def load_edge(self, img, index, mask):
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

      # external
      else:
          imgh, imgw = img.shape[0:2]
          edge = imread(self.edge_data[index])
          edge = self.resize(edge, imgh, imgw)

          # non-max suppression
          if self.nms == 1:
              edge = edge * canny(img, sigma=sigma, mask=mask)

          return edge

  def load_mask(self, img, index):
      imgh, imgw = img.shape[0:2]
      mask_type = self.mask
      
      height = random.randint(64,128)
      width = random.randint(64,128)

      # random block
      if mask_type == 1:
          return create_mask(imgw, imgh, height, width)

      elif mask_type == 2:
          percent = random.randint(10,90)
          return salt_and_pepper(percent,imgw,imgh)

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

dataset1 = dset.ImageFolder('satellite_imagery')

train_size = int(0.8 * len(dataset1))
val_size = len(dataset1) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset1, [train_size, val_size])


train_loader = torch.utils.data.DataLoader(Dataset(train_dataset), batch_size = 4, shuffle = True)
val_loader = torch.utils.data.DataLoader(Dataset(val_dataset, augment = True, training = False), batch_size = 1, shuffle = False)


def gradient_penalty(image, generated_data, gamma, discriminator):
    batch_size = image.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1)
    epsilon = epsilon.expand_as(image)

    epsilon = epsilon.cuda()

    interpolation = epsilon * image.data + (1 - epsilon) * generated_data.data
    interpolation = Variable(interpolation, requires_grad=True)

    interpolation = interpolation.cuda()

    interpolation_logits = discriminator(interpolation)
    grad_outputs = torch.ones(interpolation_logits.size())

    grad_outputs = grad_outputs.cuda()

    gradients = autograd.grad(outputs=interpolation_logits,
                              inputs=interpolation,
                              grad_outputs=grad_outputs,
                              create_graph=True,
                              retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return gamma * ((gradients_norm - 1) ** 2).mean()
    


      

      

def train(inpaint_generator, generator, discriminator, vgg19, g_optimizer, d_optimizer, adversarial_loss, l1_loss, dataloader):
  inpaint_generator.eval()
  generator.train()
  discriminator.train()

  for i, (images, images_gray, edges, masks) in enumerate(dataloader):
    if i < 4000:
      images, images_gray, edges, masks = Variable(images.cuda()), Variable(images_gray.cuda()), Variable(edges.cuda()), Variable(masks.cuda())
      batch_size = images.size(0)

  #    edges_masked = edges * (1 - masks)
   #   images_gray_masked = images_gray * (1 - masks) + masks
   #   inputs = torch.cat((images_gray_masked, edges_masked, masks), dim = 1)
   #   e_outputs = edge_generator(inputs)

   #   e_outputs = e_outputs * masks + edges * (1 - masks)

      images_masked = (images * (1 - masks).float()) + masks
      inputs = images_masked
      outputs = inpaint_generator(inputs)

      generated_data = (outputs * masks) + (images * (1 - masks))


      g_optimizer.zero_grad()
      d_optimizer.zero_grad()

      inputs = torch.cat((images_masked, generated_data), dim = 1)
      outputs = generator(inputs)

      target_real_label = 1.0
      target_fake_label = 0.0
      real_label = torch.tensor(target_real_label)
      fake_label = torch.tensor(target_fake_label)
  
      i_gen_loss = 0
      i_dis_loss = 0

      i_dis_input_real = images
      i_dis_input_fake = outputs.detach()
      i_dis_real = discriminator(i_dis_input_real)
      i_dis_fake = discriminator(i_dis_input_fake)

      i_dis_labels = real_label.expand_as(i_dis_real)
      i_dis_real_loss = adversarial_loss(i_dis_real, i_dis_labels.cuda())
      
      i_labels = fake_label.expand_as(i_dis_fake)
      i_dis_fake_loss = adversarial_loss(i_dis_fake, i_labels.cuda())

      i_dis_loss += (i_dis_real_loss + i_dis_fake_loss) / 2




      ### inpaint generator adversarial loss ###
      i_gen_input_fake = outputs
      i_gen_fake = discriminator(i_gen_input_fake)

      i_gen_labels = real_label.expand_as(i_gen_fake)
      i_gen_gan_loss = adversarial_loss(i_gen_fake, i_gen_labels.cuda()) * 0.1
      i_gen_loss += i_gen_gan_loss




      ### inpaint generator l1 loss ###
      i_gen_l1_loss = l1_loss(images, outputs) / torch.mean(masks)
      i_gen_loss += i_gen_l1_loss




      ### inpaint generator perceptual loss ###
      x_p_vgg, y_p_vgg = vgg19(outputs), vgg19(images)

      content_loss = 0.0
      content_loss += l1_loss(x_p_vgg['relu1_1'], y_p_vgg['relu1_1'])
      content_loss += l1_loss(x_p_vgg['relu2_1'], y_p_vgg['relu2_1'])
      content_loss += l1_loss(x_p_vgg['relu3_1'], y_p_vgg['relu3_1'])
      content_loss += l1_loss(x_p_vgg['relu4_1'], y_p_vgg['relu4_1'])
      content_loss += l1_loss(x_p_vgg['relu5_1'], y_p_vgg['relu5_1'])

      content_loss = content_loss * 0.1
      i_gen_loss += content_loss 





      ### inpaint generator style loss ###
      x_vgg, y_vgg = vgg19(outputs * masks), vgg19(images * masks)

      # Compute loss
      style_loss = 0.0
      style_loss += l1_loss(compute_gram(x_vgg['relu2_2']), compute_gram(y_vgg['relu2_2']))
      style_loss += l1_loss(compute_gram(x_vgg['relu3_4']), compute_gram(y_vgg['relu3_4']))
      style_loss += l1_loss(compute_gram(x_vgg['relu4_4']), compute_gram(y_vgg['relu4_4']))
      style_loss += l1_loss(compute_gram(x_vgg['relu5_2']), compute_gram(y_vgg['relu5_2']))

      style_loss = style_loss * 250
      i_gen_loss += style_loss 

      inpaint_loss.append(i_gen_loss)

      i_gen_loss.backward()
      g_optimizer.step()

      i_dis_loss.backward()
      d_optimizer.step()

      print('Epoch : %d/%d \t  Iters : %d/99  \t Discriminator Loss : %.4f \t Generator Loss : %.4f' % (epoch + 1, num_epochs, i , i_dis_loss, i_gen_loss))


    else:
      break



def validation(inpaint_generator, generator, discriminator, vgg19, adversarial_loss, l1_loss, dataloader):
#  edge_generator.eval()
  inpaint_generator.eval()
  generator.eval()
  discriminator.eval()
 
  global best_loss
  val_loss.append(0)
  
  for i,items in enumerate(dataloader):
    images, images_gray, edges, masks = items
    images, images_gray, edges, masks = images.cuda(), images_gray.cuda(), edges.cuda(), masks.cuda()
    
#    edges_masked = edges * (1 - masks)
#    images_gray_masked = images_gray * (1 - masks) + masks
#    inputs = torch.cat((images_gray_masked, edges_masked, masks), dim = 1)
#    e_outputs = edge_generator(inputs)
#    e_outputs = e_outputs * masks + edges * (1 - masks)

    images_masked = (images * (1 - masks).float()) + masks
#    inputs = torch.cat((images_masked, e_outputs), dim = 1)
    inputs = images_masked
    outputs = inpaint_generator(inputs)

    generated_data = (outputs * masks) + (images * (1 - masks))
    generated_data = torch.cat((images_masked, generated_data), dim = 1)
    outputs = generator(generated_data)


    target_real_label = 1.0
    target_fake_label = 0.0
    real_label = torch.tensor(target_real_label)
    fake_label = torch.tensor(target_fake_label)
  
    i_gen_loss = 0
    i_dis_loss = 0

    i_dis_input_real = images
    i_dis_input_fake = outputs.detach()
    i_dis_real = discriminator(i_dis_input_real)
    i_dis_fake = discriminator(i_dis_input_fake)

    i_dis_labels = real_label.expand_as(i_dis_real)
    i_dis_real_loss = adversarial_loss(i_dis_real, i_dis_labels.cuda())
      
    i_labels = fake_label.expand_as(i_dis_fake)
    i_dis_fake_loss = adversarial_loss(i_dis_fake, i_labels.cuda())

    i_dis_loss += (i_dis_real_loss + i_dis_fake_loss) / 2
  



      ### inpaint generator adversarial loss ###
    i_gen_input_fake = outputs
    i_gen_fake = discriminator(i_gen_input_fake)

    i_gen_labels = real_label.expand_as(i_gen_fake)
    i_gen_gan_loss = adversarial_loss(i_gen_fake, i_gen_labels.cuda()) * 0.1
    i_gen_loss += i_gen_gan_loss




      ### inpaint generator l1 loss ###
    i_gen_l1_loss = l1_loss(images, outputs) / torch.mean(masks)
    i_gen_loss += i_gen_l1_loss




      ### inpaint generator perceptual loss ###
    x_p_vgg, y_p_vgg = vgg19(outputs), vgg19(images)

    content_loss = 0.0
    content_loss += l1_loss(x_p_vgg['relu1_1'], y_p_vgg['relu1_1'])
    content_loss += l1_loss(x_p_vgg['relu2_1'], y_p_vgg['relu2_1'])
    content_loss += l1_loss(x_p_vgg['relu3_1'], y_p_vgg['relu3_1'])
    content_loss += l1_loss(x_p_vgg['relu4_1'], y_p_vgg['relu4_1'])
    content_loss += l1_loss(x_p_vgg['relu5_1'], y_p_vgg['relu5_1'])

    content_loss = content_loss * 0.1
    i_gen_loss += content_loss 





      ### inpaint generator style loss ###
    x_vgg, y_vgg = vgg19(outputs * masks), vgg19(images * masks)

      # Compute loss
    style_loss = 0.0
    style_loss += l1_loss(compute_gram(x_vgg['relu2_2']), compute_gram(y_vgg['relu2_2']))
    style_loss += l1_loss(compute_gram(x_vgg['relu3_4']), compute_gram(y_vgg['relu3_4']))
    style_loss += l1_loss(compute_gram(x_vgg['relu4_4']), compute_gram(y_vgg['relu4_4']))
    style_loss += l1_loss(compute_gram(x_vgg['relu5_2']), compute_gram(y_vgg['relu5_2']))

    style_loss = style_loss * 250
    i_gen_loss += style_loss 



    val_loss[-1] = val_loss[-1] + i_gen_loss.data
    print('Val_loss = %.4f' % (val_loss[-1]/(i+1)))

  val_loss[-1] = val_loss[-1]/len(val_loader)

  if best_loss > val_loss[-1]:
    best_loss = val_loss[-1]
    print('Saving...')

    state = {'generator' : generator, 'discriminator' : discriminator}
    torch.save(state, 'global_gan_imagery_salt_best.ckpt.t7')
    
    f = open("best_loss.txt", "w")
    f.write(str(best_loss.item()))
    f.close()

inpaint_loss = []

val_loss = []
torch.cuda.empty_cache()

checkpoints = torch.load('satellite_edge_connect_imagery_no_edge_salt_load.ckpt.t7')
inpaint_generator = checkpoints['inpaint_generator']
#edge_generator = checkpoints['edge_generator']

checkpoints = torch.load('global_gan_random_imagery_salt_load.ckpt.t7')
generator = checkpoints['generator']
discriminator = checkpoints['discriminator']

#generator = final_generator().cuda()
#discriminator = final_discriminator(3).cuda()

vgg19 = VGG19().cuda()

g_optimizer = optim.Adam(params = generator.parameters(), lr = float(0.0001), betas = (0.0, 0.9))
d_optimizer = optim.Adam(params = discriminator.parameters(), lr = float(0.00001), betas = (0.0, 0.9))

l1_loss = nn.L1Loss()
adversarial_loss = nn.BCELoss()

#num_epochs = 1
num_epochs = 50000000

h = open('best_loss.txt','r')
best_loss = float(h.read())
h.close()

#best_loss = 1e5

for epoch in range(num_epochs):
  train(inpaint_generator, generator, discriminator, vgg19, g_optimizer, d_optimizer, adversarial_loss, l1_loss, train_loader)  
  validation(inpaint_generator, generator, discriminator, vgg19, adversarial_loss, l1_loss, val_loader)
  print('Hidden Epoch : %d/%d' % (epoch+1, num_epochs))
  state = {'generator' : generator, 'discriminator' : discriminator}
  torch.save(state, 'global_gan_random_imagery_salt_load.ckpt.t7')

print('Epoch finally done :) ')