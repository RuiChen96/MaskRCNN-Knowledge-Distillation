# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import warnings


def bbox_transform(ex_rois, gt_rois):
    pass


def bbox_transform_inv(boxes, deltas):
    pass


def clip_boxes(boxes, im_shape):
    pass


def bbox_transform_linear(anchors, gt_boxes, alpha=10.0):
    pass


def bbox_transform_inv_linear(anchors, deltas, alpha=10.0):
    pass
