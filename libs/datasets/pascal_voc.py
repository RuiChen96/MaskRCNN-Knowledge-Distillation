from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import os.path as osp
import PIL
import cv2
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import uuid
import xml.etree.ElementTree as ET
import libs.configs.config as cfg

from .dataloader import sDataLoader
import libs.boxes.cython_bbox as cython_bbox
from libs.nets.utils import everything2tensor
from libs.layers.data_layer import data_layer_keep_aspect_ratio, \
    data_layer_keep_aspect_ratio_batch

def unique_boxes():
    pass

if __name__ == '__main__':
    cfg.data_dir = './data/pascal_voc/'
    d = pascal_voc('trainval', '0712')
    res = d.roidb
    from IPython import embed

    embed()