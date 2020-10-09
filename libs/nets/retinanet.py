from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import libs.configs.config as cfg

from .head import RetinaHead
from .model import detection_model
from .pyramid2 import PyramidFPN
from .focal_loss import FocalLoss
from .smooth_l1_loss import smooth_l1_loss
from libs.nets.utils import everything2cuda
from libs.layers.data_layer import compute_rpn_targets_in_batch
from . import utils


class RetinaNet(detection_model):
    pass
