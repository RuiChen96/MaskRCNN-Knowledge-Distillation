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
    print('Restoring from {:s} ...'.format(cfg.restore))
    meta = load_net(cfg.restore, model)
    print(meta)
    if meta[0] >= 0 and not cfg.start_over:
        start_epoch = meta[0] + 1
        lr = meta[1]
    print('Restored from {:s}, starting from {:d} epoch, lr: {:.6f}'.format(cfg.restore, start_epoch, lr))

trainable_vars = [param for param in model.parameters() if param.requires_grad]

for k, var in dict(model.named_parameters()).items():
    if var.requires_grad:
        print('gradients --- ', k)

if cfg.solver == 'SGD':
    optimizer = torch.optim.SGD()
elif cfg.solver == 'RMS':
    optimizer = torch.optim.RMSprop()
elif cfg.solver == 'Adam':
    optimizer = torch.optim.Adam()
else:
    optimizer = ''
    raise ValueError()

model.cuda()

# # DATA LOADER
get_loader = get_data_loader(cfg.datasetname)
train_data = get_loader()
class_names = train_data.dataset.classes
print('dataset len: {}'.format(len(train_data.dataset)))

tb_dir = os.path.join(
    cfg.train_dir, cfg.backbone + '_' + cfg.datasetname, time.strftime("%h%d_%H")
)
writer = tbx.FileWriter(tb_dir)
summary_out = []

global_step = 0
timer = Timer()

for ep in range(start_epoch, cfg.max_epoch):
    if ep in cfg.lr_decay_epoches and cfg.solver == 'SGD':
        lr *= cfg.lr_decay
        adjust_learning_rate(optimizer, lr)
        print('adjusting learning rate {:.6f}'.format(lr))

    for step, batch in enumerate(train_data):
        timer.tic()

        input, anchors_np, im_scale_list, image_ids, gt_boxes_list, rpn_targets, _, _ = batch
        #
        gt_boxes_list = ScatterList(gt_boxes_list)
        input = everything2cuda(input)
        rpn_targets = everything2cuda(rpn_targets)
        #
        outs = model(input, gt_boxes_list, anchors_np, rpn_targets=rpn_targets)

        if cfg.model_type == 'maskrcnn':
            rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts, anchors, \
            rois, roi_img_ids, rcnn_logit, rcnn_box, rcnn_prob, rcnn_labels, rcnn_bbtargets, rcnn_bbwghts = outs
            #
            outputs = []
            #
            targets = []
        elif cfg.model_type == 'retinanet':
            # Thinking like this: single-stage detector take rpn results as final results
            rpn_logit, rpn_box, rpn_prob, rpn_labels, rpn_bbtargets, rpn_bbwghts = outs
            #
            outputs = []
            #
            targets = []
        else:
            raise ValueError('Unknown model type: {:s}'.format(cfg.model_type))

        # # BUILD LOSS
        loss_dict = model_ori.build_losses(outputs, targets)
        loss = model_ori.loss()

        # # BACK PROPAGATION
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t = timer.toc()

        # # TENSORBOARD VISUALIZATION
        if step % cfg.display == 0:
            loss_str = ', '.join('{:s}: {:.3f}'.format(k, v) for k, v in loss_dict.iteritems())
            print(time.strftime())



if __name__ == '__main__':
    pass