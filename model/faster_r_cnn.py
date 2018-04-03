import torch
import torchvision
import numpy as np

from torch import nn
from torch.nn import functional as F

class FasterRCNN(nn.Module):

    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean = (0., 0., 0., 0.),
                 loc_noemalize_std = (0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and standard deviation
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_noemalize_std = loc_noemalize_std
        self.use_preset('evaluate')

    @property
    def num_class(self):

        # number of classes, including the background.
        return self.head.num_class

    def forward(self, x, scale = 1.):

        # x is input image, I guess
        img_size = x.shape[2:]
        # h is feature map, I guess
        h = self.extractor(x)
        # rpn receive [h, img_size and scale]
        # rpn produce [rpn_locs, rpn_scores, rois, roi_indices, anchor]
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        # head receive [h, rois (from rpn), roi_indices (from rpn)]
        # head produce [roi_cls_locs, roi_scores]
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)

        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):

        if preset == 'visualize':
            self.nms_threshold = 0.3
            self.score_threshold = 0.7
        elif preset == 'evaluate':
            self.nms_threshold = 0.3
            self.score_threshold = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0, because 0 stands for background class.
        for l in range(1, self.num_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.num_class, 4))[:, l, :]