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

def log_images(imgs, names, global_step, prefix='Image'):
    summary = []
    for i, I in enumerate(imgs):
        summary.append(tbx.summary.image('%d/%s-%s'%(global_step, prefix, names[i]), I, dataformats='HWC'))
    return summary

args = parse_args()


# config model and lr
num_anchors = len(cfg.anchor_ratios) * len(cfg.anchor_scales[0]) * len(cfg.anchor_shift) \
    if isinstance(cfg.anchor_scales[0], list) else \
    len(cfg.anchor_ratios) * len(cfg.anchor_scales)

resnet = resnet50 if cfg.backbone == 'resnet50' else resnet101
resnet = resnet18 if cfg.backbone == 'resnet18' else resnet

detection_model = MaskRCNN if cfg.model_type.lower() == 'maskrcnn' else RetinaNet

model = detection_model(resnet())

# for multiple GPUs
if torch.cuda.device_count() > 1:
    print()
    print('--- Using %d GPUs ---' % torch.cuda.device_count())
    print()
    model_ori = model
    #
    model = ListDataParallel(model)
    # batch_size * num_of_GPUs
    cfg.batch_size = cfg.batch_size * torch.cuda.device_count()
    # data_workers * num_of_GPUs
    cfg.data_workers = cfg.data_workers * torch.cuda.device_count()
    # TODO: why learning rate * num_of_GPUs ?
    cfg.lr = cfg.lr * torch.cuda.device_count()
else:
    # for single GPU
    model_ori = model

lr = cfg.lr
start_epoch = 0
if cfg.restore is not None:
    print('')


if __name__ == '__main__':
    pass