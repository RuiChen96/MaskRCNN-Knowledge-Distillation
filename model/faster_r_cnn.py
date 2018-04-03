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

    def forward(self, *input):
        