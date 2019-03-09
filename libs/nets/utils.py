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


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

