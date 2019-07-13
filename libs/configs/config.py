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
lr_decay_epoches = [10, 20, 25]
lr_decay = 0.1
lr = 0.001

use_focal_loss = True

solver = 'SGD'

"""NETWORK"""
zoom = 0
maxpool5 = True
model_type = 'maskrcnn' # or retinanet
backbone = 'resnet50'

# For COCO dataset
num_classes = 81

with_segment = True
# class activation, softmax or sigmoid?
# There is no background class for sigmoid.
class_activation = 'sigmoid'
save_prefix = ''

"""DATA"""
# support pascal_voc for now
data_workers = 4
batch_size = 6

# using which layers
strides = (8, 16, 32, 64, 128)
f_keys = ['C3', 'C4', 'C5', 'C6', 'C7']
in_channels = [512, 1024, 2048, 256, 256]

num_channels = 256
use_augment = False
training_scale = [0.3, 0.5, 0.7, 1.0]

# only for citypersons
use_extend = False

"""Anchors"""
# used to assign boxes to pyramid layers, corresponding to input size
base_size = 256
rpn_bg_threshold = 0.4
rpn_fg_threshold = 0.6
rpn_batch_size = 384
rpn_fg_fraction = 0.25
#
rpn_clobber_positives = True
# 'simple', '', 'advanced'
rpn_sample_strategy = 'simple'
# 'linear', 'fastrcnn'
rpn_box_encoding = 'fastrcnn'

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

"""LOGs"""
train_dir = './output'
# support pascal_voc
datasetname = 'pascal_voc'
tensorboard = True
