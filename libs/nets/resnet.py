from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import math
import os
import torch
import torch.utils.model_zoo as model_zoo
from . import utils
import torchvision.models.resnet


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', \
           'resnet101', 'resnet152']


model_urls = {
    'resnet18': '',
    'resnet34': '',
    'resnet50': '',
    'resnet101': '',
    'resnet152': '',
}


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.
    """
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, \
                     kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, dilation=1):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, dilation=1):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, \
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #
        padding = 1 if dilation == 1 else 2
        #
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, \
                               kernel_size=3, stride=stride, padding=padding, \
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        #
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * 4, \
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        #
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):

        self.in_planes = 64

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, \
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer()

    def _make_layer(self):
        pass

