from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class FocalLoss(_WeightedLoss):
    """"""
    def __init__(self, weight=None, size_average=True, ignore_index=-100, gamma=2, alpha=0.25, activation='sigmoid'):

        super(FocalLoss, self).__init__(weight)
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.alpha = alpha
        self.activation = activation
        self.size_average = size_average

    def softmax_loss(self, input, target):
        assert not target.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \ 
        "mark these variables as volatile or not requiring gradients"

        fg_nums = target.data.gt(0).sum.item()

