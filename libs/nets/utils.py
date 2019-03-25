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


# def adjust_learning_rate(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


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


def everything2cuda(x, volatile=False):
    if isinstance(x, np.ndarray):
        # ndarray --> tensor --> cuda
        return torch.from_numpy(x).cuda()
    elif isinstance(x, torch.FloatTensor) or \
            isinstance(x, torch.LongTensor) or \
            isinstance(x, torch.IntTensor) or \
            isinstance(x, torch.DoubleTensor):
        return x.cuda()
    elif isinstance(x, Variable):
        return x.cuda()
    elif isinstance(x, list) or isinstance(x, tuple):
        y = list()
        for i, e in enumerate(x):
            y.append(everything2cuda(e))
        # end_for
        return y
    else:
        print('Unknown data type: ', type(x))
        raise TypeError('Unknown data type, when converting to "CUDA".')


def everything2tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, Variable):
        if x.is_cuda:
            return x.cpu().data
        return x.data
    elif isinstance(x, list) or isinstance(x, tuple):
        y = list()
        for i, e in enumerate(x):
            y.append(everything2tensor(e))
        # end_for
        return y
    else:
        print('Unknown data type: ', type(x))
        raise TypeError('Unknown data type, when converting to "Torch.Tensor".')


def everything2numpy(x):
    if isinstance(x, torch.FloatTensor) or \
            isinstance(x, torch.IntTensor) or \
            isinstance(x, torch.DoubleTensor) or \
            isinstance(x, torch.LongTensor):
        return x.numpy().copy()
    if isinstance(x, Variable):
        if x.is_cuda:
            return x.cpu().data.numpy()
        return x.data.numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        y = list()
        for i, e in enumerate(x):
            y.append(tensor2numpy(e))
        # end_for
        return y
    else:
        print('Unknown data type: ', type(x))
        raise TypeError('Unknown data type, when converting to "Numpy".')


tensor2numpy = everything2numpy


def everything2cpu(x):
    if isinstance(x, np.ndarray):
        # NOTE: !!! requires_grad == False !!!
        return Variable(torch.from_numpy(x), requires_grad=False).cpu()
    elif isinstance(x, Variable):
        return x.cpu()
    elif isinstance(x, list) or isinstance(x, tuple):
        y = list()
        for i, e in enumerate(x):
            y.append(everything2cpu(e))
        # end_for
        return y
    else:
        print('Unknown data type: ', type(x))
        raise TypeError('Unknown data type, when converting to "CPU Computing".')


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_net():
    pass


def load_net():
    pass

