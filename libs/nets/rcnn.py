from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from . import utils
from math import log


class RCNN(nn.Module):

    def __init__(self, num_channels, num_classes, \
                 feat_height, feat_width, activation='softmax'):

        super(RCNN, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.trans = nn.Sequential(
            nn.Linear(in_features=num_channels * feat_height * feat_width, \
                      out_features=1024),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=False),
        )

        self.cls_out = nn.Linear(in_features=1024, out_features=num_classes)
        self.box_out = nn.Linear(in_features=1024, out_features=4)

        utils.init_gauss(self.trans, 0.01)
        utils.init_gauss(self.cls_out, 0.01)
        utils.init_gauss(self.box_out, 0.001)

        if activation == 'sigmoid':
            self.cls_out.bias.data[:] = -2.19
        elif activation == 'softmax':
            self.cls_out.bias.data[:] = 0
            self.box_out.bias.data[0::num_classes] = log(9 * (num_classes - 1))
        else:
            raise ValueError('Unknown activation function {:s}'.format(activation))

    def forward(self, pooled_features):

        x = pooled_features.view(pooled_features.size(0), -1)
        x = self.trans(x)

        cls = self.cls_out(x)
        box = self.box_out(x)

        return [cls, box]

