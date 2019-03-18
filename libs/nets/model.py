from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboardX as tbx

import libs.configs.config as cfg

from .focal_loss import FocalLoss
from .smooth_l1_loss import smooth_l1_loss
from libs.layers.box import decoding_box, apply_nms
from libs.nets.utils import everything2numpy, everything2cuda


class detection_model(nn.Module):
    """
    This module apply backbone network, build a pyramid,
    then add rpns for all layers in the pyramid.
    """
    def __init__(self, backbone, num_classes, num_anchors, is_training=True, maxpool5=True):

        super(detection_model, self).__init__()

        self.backbone = backbone
        # number of classes for rpn
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.is_training = is_training
        self.rpn_activation = cfg.class_activation

        self.rpn_outs = []
        self.loss_dict = []

        self.with_segment = cfg.with_segment

        self._score_summaries = {}
        self._hist_summaries = {}
        self.global_step = 0
        # Anchors must be set via running setup().
        self.anchors = None

        self.maxpool5 = maxpool5

        if is_training:
            # Treat rpn as a single-stage fg/bg detector.
            self.rpn_cls_loss_func = FocalLoss(gamma=2, alpha=0.25, \
                                               activation=self.rpn_activation) \
                if cfg.use_focal_loss else nn.CrossEntropyLoss()

    def forward(self, input, gt_boxes_list, anchors_np):
        # Save for class-MaskRCNN or class-RetinaNet,
        # they will "super" class-detection_model later.
        pass

    def _objectness(self):
        pass

    def _rerange(self):
        pass

    def _stage_one_results(self):
        pass

    def _thresholding(self):
        pass

    def build_losses_rpn(self):
        pass

