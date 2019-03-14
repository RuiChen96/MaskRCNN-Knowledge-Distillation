#
# This is a copy of torchvision.models.resnet.py
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import math
import os
import torch
from . import utils
import torchvision.models.resnet


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', \
           'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.
    """
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, \
                     kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution
    """
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, \
                     kernel_size=1, stride=stride, bias=False)


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

        # conv1x1()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, \
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        padding = 1 if dilation == 1 else 2

        # conv3x3()
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, \
                               kernel_size=3, stride=stride, padding=padding, \
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        # conv1x1(), out_channels = planes * self.expansion
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * 4, \
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

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

        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, \
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.maxpool5 = kwargs['maxpool5'] if 'maxpool5' in kwargs else True
        if self.maxpool5:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        else:
            self.layer4 = self._make_layer_no_downsample(block, 512, layers[3], stride=2)
            print('Removing subsample 5 ... using dilation.')

        self.avgpool = nn.AvgPool2d(7)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_planes, out_channels=planes * block.expansion, \
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, dilation))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_layer_no_downsample(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_planes, out_channels=planes * block.expansion, \
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, 1, downsample, dilation=stride))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):

        # For later use in " Feature Pyramid Network ".
        endpoints = dict()

        endpoints['C0'] = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        endpoints['C1'] = x
        x = self.maxpool(x)
        x = self.layer1(x)

        endpoints['C2'] = x
        x = self.layer2(x)

        endpoints['C3'] = x
        x = self.layer3(x)

        endpoints['C4'] = x
        x = self.layer4(x)

        endpoints['C5'] = x

        return endpoints


def resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

    return model


def resnet50(pretrained=False, weight_path=None, **kwargs):
    """"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)

    return model


def resnet101(pretrained=False, **kwargs):
    """"""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)

    return model


# Not frequently USED Model
def resnet34(pretrained=False, **kwargs):
    """"""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)

    return model


def resnet152(pretrained=False, **kwargs):
    """"""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)

    return model

