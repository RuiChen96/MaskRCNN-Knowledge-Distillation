#
# This is based on anchor_target_layer.py from "py-faster-rcnn".
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import libs.boxes.cython_bbox as cython_bbox
import libs.configs.config as cfg

_DEBUG = False


def encode(gt_boxes, all_anchors):
    """

    :param gt_boxes: an array of shape (G x 5), [x1, y1, x2, y2, class]
    :param all_anchors: an array of shape (h, w, A, 4)
    :return: labels: (N x 1) array in [-1, num_classes], negative labels are ignored
    bbox_targets: (N x 4) regression targets
    bbox_inside_weights: (N x 4), in {0, 1} indicating to which class is assigned
    """

    all_anchors = all_anchors.reshape([-1, 4])
    anchors = all_anchors
    total_anchors = all_anchors.shape[0]
    bbox_flags_ = np.zeros([total_anchors], dtype=np.int32)

    if gt_boxes.size > 0:
        overlaps = cython_bbox.bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float)
        )

        # (A)
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(total_anchors), gt_assignment]

        # (G)
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        # Add Mask.
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # 0 - background, 1 - foreground, -1 - ignore
        labels = gt_boxes[gt_assignment, 4]
        labels[max_overlaps < cfg.rpn_bg_threshold] = 0
        # ignore rpn_bg_threshold <= max_overlaps < rpn_fg_threshold
        labels[np.logical_and(max_overlaps < cfg.rpn_fg_threshold,
                              max_overlaps >= cfg.rpn_bg_threshold)] = -1
        bbox_flags_[max_overlaps >= 0.5] = 1

        labels[gt_argmax_overlaps] = gt_boxes[gt_assignment[gt_argmax_overlaps], 4]

        if cfg.rpn_clobber_positives:
            labels[max_overlaps < cfg.rpn_bg_threshold] = 0
        bbox_flags_[labels >= 1] = 1

        if _DEBUG:
            pass

        ignored_inds = np.where(gt_boxes[:, -1] < 0)[0]
        if ignored_inds.size > 0:
            ignored_areas = gt_boxes[ignored_inds, :]
            intersecs = cython_bbox.bbox_intersections(
                np.ascontiguousarray(),
                np.ascontiguousarray()
            )

