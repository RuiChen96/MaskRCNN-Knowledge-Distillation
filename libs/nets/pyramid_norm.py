#
# Enhanced Feature Pyramid Networks by "Pyramid Normalization"
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class PyramidNorm(nn.Module):
    """
    Assuming we have already got a meta-feature,
    which is a concatenated result of several pyramid features
    along the channel dimension. However, it is un-normalized.
    This class will normalize this feature using BatchNorm
    along the ***Pyramid*** dimension.
    """
    def __init__(self, groups, eps=1e-5, momentum=0.1, affine=True):
        super(PyramidNorm, self).__init__()
        self.groups = groups
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(groups))
            self.bias = Parameter(torch.Tensor(groups))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(groups))
        self.register_buffer('runnig_var', torch.ones(groups))

        # reset params
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

        self.train()

    def forward(self, input):
        # input of shape (N, C, H, W), reshape to (N, g, wH, W), g*w == C
        # do a batch norm
        # reshape back to (N, C, H, W)
        n, c, h, w = input.size()
        normed = F.batch_norm(input.view(n, self.groups, -1, w),
                              self.running_mean, self.running_var, self.weight, self.bias,
                              self.training, self.momentum, self.eps)

        return normed.view(n, c, h, w)

    def __repr__(self):
        return ('{name}({groups}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

