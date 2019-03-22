from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import libs.configs.config as cfg
# from


class PyramidRoIAlign2(nn.Module):

    def __init__(self, aligned_height, aligned_width, num_channels=256):

        super(PyramidRoIAlign2, self).__init__()

        self.num_levels = len(cfg.strides)
        self.aligned_height = aligned_height
        self.aligned_width = aligned_width
        # TODO: RoIAlign
        self.roi_align = []
        self.spatial_scales = np.asarray([])

        # Concatenate
        self.reduce = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(inplace=True)
        )

    def forward(self, pyramids, bboxes, batch_inds):
        rois = torch.cat()
        rois = rois.detach()

        res = []
        for level in range(self.num_levels):
            f = self.roi_align()
            res.append(f)

        aligned_features = torch.cat()
        aligned_features = self.reduce(aligned_features)

        return aligned_features


if __name__ == '__main__':
    pass

