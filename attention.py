from init_weights import *
import random
import math
from random import randint
import torch
import torch.nn as nn
from torch.nn import init
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


class CBAM(nn.Module):

    def __init__(self, n_channels_in):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in

        self.channel_attention = ChannelAttention(n_channels_in, 1)
        self.spatial_attention1 = SpatialAttention(3)
        self.spatial_attention2 = SpatialAttention(7)
        self.spatial_attention3 = SpatialAttention(11)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att1 = self.spatial_attention1(fp)
        spat_att2 = self.spatial_attention2(fp)
        spat_att3 = self.spatial_attention3(fp)
        # print(spat_att.size())
        spat_att = torch.max(spat_att1, spat_att2)
        spat_att = torch.max(spat_att3, spat_att)
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

    def forward(self, x):
        x = x.contiguous()
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        x = x.contiguous()
        if pool == "max":
            x = TF.max_pool1d(x, c)
        elif pool == "avg":
            x = TF.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.contiguous()
        x = x.view(b,1,h,w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = self.n_channels_in

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = TF.avg_pool2d(x, kernel )
        max_pool = TF.max_pool2d(x, kernel)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out



### Attention Gated Models ###

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
        phi_g = TF.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = TF.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = TF.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = TF.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
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
