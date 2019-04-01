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

        self.rpn_outs = list()
        self.loss_dict = list()

        self.with_segment = cfg.with_segment

        self._score_summaries = dict()
        self._hist_summaries = dict()
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
        #
        rpn_logit = [rpn[0].view(n, c, -1) for rpn in rpn_outs]
        rpn_box = [rpn[1].view(n, cb, -1) for rpn in rpn_outs]
        #
        rpn_logit = torch.cat(rpn_logit, dim=2)
        rpn_box = torch.cat(rpn_box, dim=2)
        #
        rpn_logit = rpn_logit.permute(0, 2, 1).contiguous().view(-1, last_dimension)
        num_endpoints = rpn_logit.size()[0]
        rpn_box = rpn_box.permute(0, 2, 1).contiguous().view(num_endpoints, -1)

        return rpn_logit, rpn_box

    def _stage_one_results(self, rpn_box, rpn_prob, anchors, top_n=2000, \
                           overlap_threshold=0.7, \
                           top_n_post_nms=None):
        boxes, probs, img_ids, anchors = \
            self._decode_and_choose_top_n_stage1(rpn_box, rpn_prob, anchors, top_n=top_n)

    def _thresholding(self):
        pass

    def build_losses_rpn(self, rpn_logits, rpn_box, rpn_prob, \
                         rpn_labels, rpn_bboxes, rpn_bbwghts):
        """
        With OHEM (Online Hard Example Mining)
        """
        # TODO:
        rpn_cls_loss = []
        rpn_box_loss = []

        return rpn_cls_loss, rpn_box_loss

    def build_losses_rpn_faster_rcnn(self, rpn_logits, rpn_box, rpn_prob, \
                                     rpn_labels, rpn_bboxes, rpn_bbwghts):
        """
        Without OHEM (Online Hard Example Mining)
        """
        # TODO:
        rpn_cls_loss = []
        rpn_box_loss = []

        return rpn_cls_loss, rpn_box_loss

    def build_losses(self, outputs, targets):
        pass

    def loss(self):
        pass

    def cls_loss(self):
        # Treats RPN as a single-stage object detector.
        return self.loss_dict['rpn_cls_loss']

    def box_loss(self):
        return self.loss_dict['rpn_box_loss']

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

    def _decoding_and_thresholding_stage1(self, rpn_box, rpn_prob, anchors, \
                                          score_threshold=0.3, max_dets=100):
        selected_boxes, selected_probs, selected_img_ids, selected_anchors = \
            self._decode_and_choose_top_n_stage1(rpn_box, rpn_prob, anchors, top_n=max_dets * 3)

        objness = self._objectness(selected_probs)
        inds = objness.data.ge(score_threshold).nonzero().view(-1)

        if inds.numel() == 0:
            _, inds = objness.sort(dim=0, descending=True)
            inds = inds[:1]

        selected_boxes = selected_boxes[inds]
        selected_probs = selected_probs[inds]
        selected_img_ids = selected_img_ids[inds]
        selected_anchors = selected_anchors[inds]

        return selected_boxes, selected_probs, selected_img_ids, selected_anchors

    @staticmethod
    def _apply_nms_in_batch(boxes, probs, img_ids, anchors, activation, overlap_threshold=0.5):
        all_keeps = list()
        return boxes[all_keeps], probs[all_keeps], img_ids[all_keeps], anchors[all_keeps]

    @staticmethod
    def to_Dets(boxes, probs, img_ids):
        """
        For each bbox, assign it with "the class" of the max prob.
        """
        boxes, probs, img_ids = everything2numpy([boxes, probs, img_ids])
        Dets = list()

        for i in range(0, cfg.batch_size):
            inds = np.where(img_ids == i)[0]
            probs_ = probs[inds]
            boxes_ = boxes[inds]
            if probs_.shape[1] == 2:
                cls_ids = np.ones((probs_.shape[0], ), dtype=np.int32)
                cls_probs = probs_[:, 1]
            else:
                cls_ids = probs_[:, 1:].argmax(axis=1) + 1
                cls_probs = probs_[np.arange(probs_.shape[0]), cls_ids]

            dets = np.concatenate(
                (boxes_.reshape(-1, 4),
                 cls_probs[:, np.newaxis],
                 cls_ids[:, np.newaxis]),
                axis=1
            )

            Dets.append(dets)
        # end_for
        return Dets

    @staticmethod
    def to_Dets_sigmoid(boxes, probs, img_ids):
        """
        For each bbox, assign the class with the max prob.
        NOTE: there is no background class,
        so the implementation is slightly different.
        """
        boxes, probs, img_ids = everything2numpy([boxes, probs, img_ids])
        Dets = list()

        for i in range(0, cfg.batch_size):
            inds = np.where(img_ids == i)[0]
            probs_ = probs[inds]
            boxes_ = boxes[inds]
            # !!! Difference is here. !!!
            if probs_.ndim == 1 or probs_.shape[1] == 1:
                cls_ids = np.ones((probs_.shape[0], ), dtype=np.int32)
                cls_probs = probs_.view(-1)
            else:
                cls_ids = probs_.argmax(axis=1) + 1
                cls_probs = probs_.max(axis=1)

            dets = np.concatenate(
                (boxes_.reshape(-1, 4),
                 cls_probs[:, np.newaxis],
                 cls_ids[:, np.newaxis]),
                axis=1
            )

            Dets.append(dets)
        # end_for
        return Dets

    @staticmethod
    def to_Dets2(boxes, probs, img_ids, score_threshold=0.1):
        """
        For each bbox, there may be more than one "class" labels.
        """
        boxes, probs, img_ids = everything2numpy([boxes, probs, img_ids])
        Dets = list()

        for i in range(0, cfg.batch_size):
            inds = np.where(img_ids == i)[0]
            probs_ = probs[inds]
            boxes_ = boxes[inds]
            if probs_.shape[1] == 2:
                cls_ids = np.ones((probs_.shape[0], ), dtype=np.int32)
                cls_probs = probs_[:, 1]
                dets = np.concatenate(
                    (boxes_.reshape(-1, 4),
                     cls_probs[:, np.newaxis],
                     cls_ids[:, np.newaxis]),
                    axis=1
                )
            else:
                d0_inds, d1_inds = np.where(probs_[:, 1:] > score_threshold)
                if d0_inds.size > 0:
                    cls_ids = d1_inds + 1
                    cls_probs = probs_[d0_inds, cls_ids]
                    boxes_ = boxes_[d0_inds, :]
                    dets = np.concatenate(
                        (boxes_.reshape(-1, 4),
                         cls_probs[:, np.newaxis],
                         cls_ids[:, np.newaxis]),
                        axis=1
                    )
                else:
                    cls_ids = probs_[:, 1:].argmax(axis=1) + 1
                    cls_probs = probs_[np.arange(probs_.shape[0]), cls_ids]
                    dets = np.concatenate(
                        (boxes_.reshape(-1, 4),
                         cls_probs[:, np.newaxis],
                         cls_ids[:, np.newaxis]),
                        axis=1
                    )
            # end if_else
            Dets.append(dets)
        # end_for
        return Dets

    @staticmethod
    def to_Dets2_sigmoid(boxes, probs, img_ids, score_threshold=0.1):
        boxes, probs, img_ids = everything2numpy([boxes, probs, img_ids])
        Dets = list()
        for i in range(0, cfg.batch_size):
            inds = np.where(img_ids == i)[0]
            probs_ = probs[inds]
            boxes_ = boxes[inds]
            if probs_.ndim == 1 or probs_.shape[1] == 1:
                cls_ids = np.ones((probs_.shape[0], ), dtype=np.int32)
                cls_probs = probs_.view(-1)
                dets = np.concatenate(
                    (boxes_.reshape(-1, 4),
                     cls_probs[:, np.newaxis],
                     cls_ids[:, np.newaxis]),
                    axis=1
                )
            else:
                d0_inds, d1_inds = np.where(probs_ > score_threshold)
                if d0_inds.size > 0:
                    cls_ids = d1_inds + 1
                    cls_probs = probs_[d0_inds, d1_inds]
                    boxes_ = boxes_[d0_inds, :]
                    dets = np.concatenate(
                        (boxes_.reshape(-1, 4),
                         cls_probs[:, np.newaxis],
                         cls_ids[:, np.newaxis]),
                        axis=1
                    )
                else:
                    cls_ids = probs_.argmax(axis=1) + 1
                    cls_probs = probs_[np.arange(probs_.shape[0]), cls_ids - 1]
                    dets = np.concatenate(
                        (boxes_.reshape(-1, 4),
                         cls_probs[:, np.newaxis],
                         cls_ids[:, np.newaxis]),
                        axis=1
                    )
            # end if_else
            Dets.append(dets)
        # end_for
        return Dets

    def get_final_results(self, outputs, anchors, **kwargs):
        pass

    def get_final_results_stage1(self, rpn_box, rpn_prob, anchors, \
                                 score_threshold=0.1, \
                                 max_dets=100, \
                                 overlap_threshold=0.5):

        selected_boxes, selected_probs, selected_img_ids, selected_anchors = \
            self._decoding_and_thresholding_stage1(rpn_box, rpn_prob, anchors, \
                                                   score_threshold=score_threshold, \
                                                   max_dets=max_dets * 3)

        selected_boxes, selected_probs, selected_img_ids, selected_anchors = \
            self._apply_nms_in_batch(selected_boxes, selected_probs, \
                                     selected_img_ids, selected_anchors, \
                                     activation=self.rpn_activation, \
                                     overlap_threshold=overlap_threshold)

        if self.rpn_activation == 'softmax':
            Dets = self.to_Dets2(selected_boxes, selected_probs, \
                                 selected_img_ids, score_threshold)
        elif self.rpn_activation == 'sigmoid':
            Dets = self.to_Dets2_sigmoid(selected_boxes, selected_probs, \
                                         selected_img_ids, score_threshold)
        else:
            raise ValueError('Unknown activation function {:s}'.format(self.rpn_activation))

        return Dets

    def get_pos_anchors(self, score_threshold=0.1, max_dets=100):
        _, selected_probs, selected_img_ids, selected_anchors = \
            self._decoding_and_thresholding_stage1(score_threshold=score_threshold, max_dets=max_dets)

        if self.rpn_activation == 'softmax':
            Dets = self.to_Dets(selected_anchors, selected_probs, selected_img_ids)
        elif self.rpn_activation == 'sigmoid':
            Dets = self.to_Dets_sigmoid(selected_anchors, selected_probs, selected_img_ids)
        else:
            raise ValueError('Unknown activation function {:s}'.format(self.rpn_activation))

        return Dets

    def _to_one_hot(self, y, num_classes):
        c = num_classes + 1 if self.rpn_activation == 'sigmoid' else num_classes
        y_ = torch.FloatTensor(y.size()[0], c).zero_()
        y_ = y_.scatter_(1, y.view(-1, 1).data.cpu(), 1.0).cuda()
        if self.rpn_activation == 'sigmoid':
            y_ = y_[:, 1:]
        if y.is_cuda:
            y_ = y_.cuda()
        return y_

    def de_frozen_backbone(self):
        self.backbone.de_frozen()

    def _add_scalar_summary(self, key, tensor):
        if isinstance(tensor, torch.Tensor):
            return tbx.summary.scalar(key + '/L1', torch.abs(tensor).mean().data.cpu().numpy())
        elif isinstance(tensor, float):
            return tbx.summary.scalar(key, tensor)

    def _add_hist_summary(self, key, tensor):
        return tbx.summary.histogram(key, tensor.data.cpu().numpy(), bins='auto')

    def get_summaries(self, is_training=True):
        summaries = list()

        for key, var in self._score_summaries.items():
            # save scalar
            summaries.append(self._add_scalar_summary(key, var))
        # re-init dict
        self._score_summaries = dict()
        # Add act summaries
        # for key, var in self._hist_summaries.items():
        #     summaries.append(self._add_hist_summary(key, var))
        self._hist_summaries = dict()
        # Add train summaries
        if is_training:
            for key, var in dict(self.named_parameters()).items():
                if var.requires_grad:
                    # summaries.append(self._add_hist_summary(key, var))
                    summaries.append(self._add_scalar_summary('Params/' + key, var))
                    summaries.append(self._add_scalar_summary('Grads/' + key, var.grad))

        return summaries

