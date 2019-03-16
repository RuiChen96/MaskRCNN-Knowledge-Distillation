from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from . import utils
from math import log


class RPNHead(nn.Module):

    def __init__(self, in_channels, num_classes, num_anchors, \
                 num_channels=256, activation='sigmoid'):

        super(RPNHead, self).__init__()

        conv_combo = utils.Conv2d
        self.conv = conv_combo(in_channels=in_channels, out_channels=num_channels, \
                               kernel_size=3, same_padding=True)
        # out_channels = num_anchors * 2 (fg / bg)
        self.conv_cls_out = nn.Conv2d(in_channels=num_channels, out_channels=num_anchors * 2, \
                                      kernel_size=1, padding=0, bias=True)
        # out_channels = num_anchors * 4 (four offsets)
        self.conv_box_out = nn.Conv2d(in_channels=num_channels, out_channels=num_anchors * 4, \
                                      kernel_size=1, padding=0, bias=True)

        # utils.init_xavier(self)
        utils.init_gauss(self.conv, std=0.01)
        utils.init_gauss(self.conv_cls_out, std=0.01)
        utils.init_gauss(self.conv_box_out, std=0.001)

        # set background prior larger than foreground
        # to prevent network blow up in the early iterations
        self.conv_cls_out.bias.data.zero_()
        if activation == 'sigmoid':
            self.conv_cls_out.bias.data[:] = -4.595
        else:
            self.conv_cls_out.bias.data[:] = 0
            self.conv_cls_out.bias.data[0::num_classes] = log(99 * (num_classes - 1))

    def forward(self, x):
        rpn_logits = self.conv(x)
        return [self.conv_cls_out(rpn_logits), self.conv_box_out(rpn_logits)]


class RetinaHead(nn.Module):
    pass

