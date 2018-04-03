import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from model.utils.bbox_tool import generate_anchor_base

class RegionProposalNetwork(nn.Module):

    def __init__(
            self, in_channels = 512, mid_channels = 512, ratios = [0.5, 1, 2],
            anchor_scales = [8, 16, 32], feat_stride = 16,
            proposal_creator_params = dict()
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base()
