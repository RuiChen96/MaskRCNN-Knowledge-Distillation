from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import time
import argparse

import torch
import torch.utils.data
import torch.nn as nn
import tensorboardX as tbx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import libs.configs.config as cfg
import libs.postprocessings.single_shot as single_shot

from libs.nets.resnet import resnet18, resnet50, resnet101
from libs.nets.MaskRCNN import MaskRCNN
from libs.nets.RetinaNet import RetinaNet
from libs.nets.data_parallel import ListDataParallel, ScatterList

from libs.datasets.factory import get_data_loader
from libs.utils.timer import Timer
from libs.nets.utils import everything2cuda, everything2numpy, \
    adjust_learning_rate, load_net, save_net

torch.backends.cudnn.benchmark = True
seed = 42
torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    pass

def log_images():
    pass


if __name__ == '__main__':
    pass