from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

"""RUN Configs"""
cuda = True
display = 10
log_image = 200
restore = None

"""TRAINING"""
start_over = False
max_epoch = 30
lr = 0.001

"""NETWORK"""
zoom = 0
maxpool5 = True
model_type = 'maskrcnn' # or retinanet
backbone = 'resnet50'

"""DATA"""
# support pascal_voc for now
data_workers = 4
batch_size = 6

"""Anchor Output"""
ANCHORS = []
anchor_scales = [2, 4, 8, 16, 32]
anchor_scales = [
    [2, 2.52, 3.17],
    [4, 5.04, 6.35],
    [8, 10.08, 12.70],
    [16, 20.16, 25.40],
    [32, 40.32, 50.80],
]
anchor_ratios = [0.5, 1.0, 2.0]
anchor_base = 16
anchor_shift = [[0.0, 0.0], ]