#
# enhanced feature pyramid networks
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class PyramidNorm(nn.Module):

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
        n, c, h, w = input.size()
        normed = F.batch_norm()

        return normed.view(n, c, h, w)

    def __repr__(self):
        return ()

