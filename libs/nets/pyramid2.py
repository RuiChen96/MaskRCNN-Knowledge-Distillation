#
# enhanced feature pyramid networks
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .pyramid_norm import PyramidNorm
import numpy as np

class PyramidFPN(nn.Module):

    def __init__(self, in_channels, f_keys, num_channels=256):

        super(PyramidFPN, self).__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels
        # only support f_keys = ['C2', 'C3', 'C4', 'C5']

        # if 'C2' in f_keys --> self.with_c2 = True
        self.with_c2 = 'C2' in f_keys

        if self.with_c2:
            # ['C2', 'C3', 'C4', 'C5']

            # For Lateral Connection
            self.lateral_c2 = nn.Conv2d()
            self.lateral_c3 = nn.Conv2d()
            self.lateral_c4 = nn.Conv2d()
            self.lateral_c5 = nn.Conv2d()

            # For Tail Connection
            self.tail_c5 = nn.Conv2d()
            self.tail_c4 = nn.Conv2d()
            self.tail_c3 = nn.Conv2d()
            self.tail_c2 = nn.Conv2d()
        else:
            # ['C3', 'C4', 'C5']

            # For Lateral Connection
            self.lateral_c3 = nn.Conv2d()
            self.lateral_c4 = nn.Conv2d()
            self.lateral_c5 = nn.Conv2d()

            # For Tail Connection
            self.tail_c5 = nn.Conv2d()
            self.tail_c4 = nn.Conv2d()
            self.tail_c3 = nn.Conv2d()
        # end if_else
        utils.init_xavier(self)

    def forward(self, *input):
        pass

