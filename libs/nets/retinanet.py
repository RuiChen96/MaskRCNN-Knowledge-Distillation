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

    def __init__(self, backbone, num_classes, num_anchors,
                 strides=[8, 16, 32, 64, 128],
                 in_channels=[512, 1024, 2048, 256, 256],
                 f_keys=['C3', 'C4', 'C5', 'C6', 'C7'],
                 num_channels=256,
                 is_training=True,
                 activation='sigmoid'):

        super(RetinaNet, self).__init__(backbone, num_classes, num_anchors, is_training=is_training)
        self.rpn_activation = activation
        self.num_classes = num_classes - 1 if self.rpn_activation == 'sigmoid' else num_classes

        assert len(strides) == len(in_channels) == len(f_keys)

        self.conv6 = nn.Conv2d(2048, num_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu6 = nn.ReLU()
        self.conv6 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu7 = nn.ReLU()
        utils.init_xavier(self.conv6)
        utils.init_xavier(self.conv7)

        self.pyramid = PyramidFPN(in_channels, f_keys, num_channels)
        self.rpn = RetinaNet(num_channels, self.num_classes, num_anchors,
                             num_channels=256, activation=self.rpn_activation)

        if is_training:
            self.rpn_cls_loss_func = FocalLoss(gamma=2, alpha=0.25, activation=self.rpn_activation) \
                if cfg.use_focal_loss else nn.CrossEntropyLoss()

    def forward(self, input, gt_boxes_list, anchors_np, rpn_targets=None):

        anchors = torch.from_numpy(anchors_np).cuda()
        endpoints = self.backbone(input)

        P6 = self.conv6(endpoints['C5'])
        P7 = self.conv7(self.relu6(P6))
        Ps = self.pyramid(endpoints)
        Ps.append(P6)
        Ps.append(P7)

        rpn_outs = []
        for i, f in enumerate(Ps):
            rpn_outs.append(self.rpn(f))
        rpn_logit, rpn_box = self._rerange(rpn_outs)
        rpn_prob = F.sigmoid(rpn_logit) if self.rpn_activation == 'sigmoid' else F.softmax(rpn_logit, dim=-1)
        rpn_prob.detach()

        if self.is_training:
            if rpn_targets is None:
                rpn_targets = compute_rpn_targets_in_batch(gt_boxes_list, anchors_np)
                rpn_labels, _, rpn_bbtargets, rpn_bbwghts = everything2cuda(rpn_targets)
            else:
                rpn_labels, rpn_bbtargets, rpn_bbwghts = rpn_targets
        else:
            rpn_labels = rpn_bbtargets = rpn_bbwghts = None

        return rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts

    def build_losses(self, outputs, targets):

        rpn_logit, rpn_box, rpn_prob = outputs[:3]
        rpn_labels, rpn_bbtargets, rpn_bbwghts = targets[:3]
        rpn_cls_loss, rpn_box_loss = self.build_losses_rpn(rpn_logit, rpn_box, rpn_prob,
                                                           rpn_labels, rpn_bbtargets, rpn_bbwghts)
        self.loss_dict = {
            'rpn_cls_loss': rpn_cls_loss, 'rpn_box_loss': rpn_box_loss,
        }
        return self.loss_dict

    def loss(self, loss_weights=None):
        return self.loss_dict['rpn_cls_loss'] + self.loss_dict['rpn_box_loss']

    def get_final_results(self, outputs, anchors,
                          score_threshold=0.1, max_dets=100, overlap_threshold=0.5):
        rpn_logit, rpn_box, rpn_prob = outputs[:3]
        Dets = self.get_final_results_stage1(rpn_box, rpn_prob, anchors,
                                             score_threshold, max_dets, overlap_threshold)
        return {'Stage1': Dets}
