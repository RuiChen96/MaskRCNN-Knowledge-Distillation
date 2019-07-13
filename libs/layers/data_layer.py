from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import libs.configs.config as cfg

from libs.preprocessings.fixed_size import preprocess_train_keep_aspect_ratio

from . import anchor_ss


def data_layer_keep_aspect_ratio(img_name, bboxes, classes, inst_masks, mask, is_training):
    """
    Returns the training labels
    1. resize image, boxes, inst_masks, mask
    2. data augmentation
    """
    im = cv2.imread(img_name).astype(np.float32)
    if im.size == im.shape[0] * im.shape[1]:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = im.astype(np.float32)
    annots = {}

    if is_training:
        im, bboxes, classes, inst_masks, mask, im_scale = \
            preprocess_train_keep_aspect_ratio()


def compute_rpn_targets_in_batch(gt_boxes_list, anchors):
    pass

