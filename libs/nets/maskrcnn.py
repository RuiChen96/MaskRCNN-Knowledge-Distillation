from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import libs.configs.config as cfg

from .head import RPNHead
from .model import detection_model
from .pyramid2 import PyramidFPN
from .rcnn import RCNN
from .focal_loss import FocalLoss, SigmoidCrossEntropy

from .smooth_l1_loss import smooth_l1_loss

from libs.layers.data_layer import compute_rpn_targets_in_batch

from libs.nets.utils import everything2cuda, everything2numpy

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

        # foreground/background classification
        self.rpn_activation = 'softmax'
        self.rpn_cls_loss_func = FocalLoss(gamma=2, alpha=0.25, \
                                           activation=self.rpn_activation) \
            if cfg.use_focal_loss else nn.CrossEntropyLoss(ignore_index=-1)
        self.rpn = RPNHead(in_channels=num_channels, num_classes=2, num_anchors=num_anchors, \
                           num_channels=256, activation=self.rpn_activation)

        # TODO: Pyramid RoIAlign 2
        self.pyramid_roi_align = []

        self.rcnn = RCNN(num_channels=num_channels, num_classes=num_classes, \
                         feat_height=7, feat_width=7, activation=self.activation)
        if self.activation == 'softmax':
            self.rcnn_cls_loss_func = nn.CrossEntropyLoss()
        elif self.activation == 'sigmoid':
            self.rcnn_cls_loss_func = SigmoidCrossEntropy()

        if is_training:
            # TODO: RoITarget
            self.roi_target = []
            # TODO: AnchorTarget
            self.anchor_target = []

    def forward(self, input, gt_boxes_list, anchors_np, rpn_targets=None):

        batch_size = input.size(0)
        # torch.from_numpy() :
        # The returned tensor and ndarray share the same memory.
        anchors = torch.from_numpy(anchors_np).cuda()
        endpoints = self.backbone(input)

        # Currently No ZoomNet.

        Ps = self.pyramid(endpoints)
        rpn_outs = []
        # f means "Floor" in Feature Pyramids.
        for i, f in enumerate(Ps):
            rpn_outs.append(self.rpn(f))

        rpn_logit, rpn_box = self._rerange(rpn_outs, last_dimension=2)
        rpn_prob = F.sigmoid(rpn_logit) if self.rpn_activation == 'sigmoid' \
            else F.softmax(rpn_logit, dim=-1)
        # This is different from "rpn_prob.detach()"
        rpn_prob = rpn_prob.detach()

        if self.is_training:
            assert input.size(0) == len(gt_boxes_list), \
                '{:d} vs {:d}'.format(input.size(0), len(gt_boxes_list))
            if rpn_targets is None:
                # TODO: compute_rpn_targets_in_batch()
                rpn_targets = compute_rpn_targets_in_batch(gt_boxes_list, anchors_np)
                rpn_labels, _, rpn_bbtargets, rpn_bbwghts = everything2cuda(rpn_targets)
            else:
                rpn_labels, rpn_bbtargets, rpn_bbwghts = rpn_targets
            # end if-else

            # TODO: _stage_one_results()
            rois, probs, roi_img_ids = self._stage_one_results(rpn_box, rpn_prob, anchors, \
                                                               top_n=20000 * batch_size, \
                                                               overlap_threshold=0.7, \
                                                               top_n_post_nms=2000)
            rois, roi_labels, roi_img_ids = sample_rois(rois, roi_img_ids, gt_boxes_list)
        else:
            rpn_labels = rpn_bbtargets = rpn_bbwghts = None
            rois, probs, roi_img_ids = self._stage_one_results(rpn_box, rpn_prob, anchors, \
                                                               top_n=6000 * batch_size, \
                                                               overlap_threshold=0.7)
            rois, probs, roi_img_ids = self._thresholding(rois, probs, roi_img_ids, 0.05)
        # end if-else

        # TODO: pyramid_roi_align()
        rcnn_feats = self.pyramid_roi_align()










