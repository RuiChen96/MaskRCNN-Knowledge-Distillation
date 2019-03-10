from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

import numpy as np

class PyramidFPN(nn.Module):

    def __init__(self, in_channels, f_keys, num_channels=256):

        super(PyramidFPN, self).__init__()

        