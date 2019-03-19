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

    def _objectness(self, probs, activation=None):
        activation = self.rpn_activation if activation is None else activation
        if activation == 'softmax':
            return 1. - probs[:, 0]
        elif activation == 'sigmoid':
            return probs.max(dim=1)[0]
        else:
            raise ValueError('Unknown activation function {:s}'.format(activation))

    def _rerange(self, rpn_outs, last_dimension=None):
        """
        Rerange outputs of shape (Pyramid, N, C, H, W) to (N x L x H x W, C)
        """
        last_dimension = self.num_classes if last_dimension is None else last_dimension
        n = rpn_outs[0][0].size()[0]
        c = rpn_outs[0][0].size()[1]
        cb = rpn_outs[0][1].size()[1]
        rpn_logit = []
        rpn_box = []

        return rpn_logit, rpn_box

    def _stage_one_results(self, rpn_box, rpn_prob, anchors, top_n=2000, \
                           overlap_threshold=0.7, \
                           top_n_post_nms=None):
        boxes, probs, img_ids, anchors = \
            self._decode_and_choose_top_n_stage1(rpn_box, rpn_prob, anchors, top_n=top_n)

    def _thresholding(self):
        pass

    def build_losses_rpn(self):
        pass

    def _decode_and_choose_top_n_stage1(self, rpn_box, rpn_prob, anchors, top_n=1000):

        objness = self._objectness(rpn_prob)
        _, inds = objness.sort(dim=0, descending=True)
        inds = inds[: top_n]

        selected_boxes = rpn_box[inds]
        selected_probs = rpn_prob[inds]
        anchor_ids = inds % anchors.size(0)
        selected_anchors = anchors[anchor_ids]
        selected_boxes = decoding_box(selected_boxes, selected_anchors, \
                                      box_encoding=cfg.rpn_box_encoding)
        selected_img_ids = inds / anchors.size(0)

        return selected_boxes, selected_probs, selected_img_ids, selected_anchors

