from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import libs.configs.config as cfg

from .pyramid2 import PyramidFPN

import time

class MaskRCNN(detection_model):
    """
    Treat RPN as a foreground/background binary classification task.
    """
    def __init__(self, backbone, num_classes, num_anchors,
                 strides=[4, 8, 16, 32],
                 in_channels=[256, 512, 1024, 2048],
                 f_keys=['C2', 'C3', 'C4', 'C5'],
                 num_channels=256,
                 is_training=True,
                 activation='sigmoid'):

        super(MaskRCNN, self).__init__(backbone, 2, num_anchors, is_training=is_training)

        assert len(strides) == len(in_channels) == len(f_keys)
        self.activation = activation
        self.num_classes = num_classes - 1 if self.activation == 'sigmoid' else num_classes

        self.maxpool5 = cfg.maxpool5
        self.pyramid = PyramidFPN(in_channels, f_keys, num_channels)