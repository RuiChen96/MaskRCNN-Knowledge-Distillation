from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import h5py
import math


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_gauss(module, std):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data.normal_(0, std)
            try:
                m.bias.data.zero_()
            except:
                print('has no bias')


def init_xavier(module):
    for m in module.modules():
        # use xavier to init nn.Conv2d.weight or nn.Linear.weight
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            print(m.weight.size(), 'xavier init')
            try:
                m.bias.data.zero_()
            except:
                print('has no bias')


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bias=True):

        super(Conv2d, self).__init__()

        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def everything2numpy(x):
    pass


def everything2cuda(x, volatile=False):
    pass

