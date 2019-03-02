from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torch

import os
from libs.datasets.imdb import imdb
import libs.datasets.ds_utils as ds_utils

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

class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None, is_training=True):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self.__classes = ('__background__', # always index 0
                          'aeroplane', 'bicycle', 'bird', 'boat',
                          'bottle', 'bus', 'car', 'cat', 'chair',
                          'cow', 'diningtable', 'dog', 'horse',
                          'motorbike', 'person', 'pottedplant',
                          'sheep', 'sofa', 'train', 'tvmonitor'
                          )
        self._class_to_ind = dict(zip(self.classes), xrange(self.num_classes))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup':       True,
                       'use_salt':      True,
                       'use_diff':      False,
                       'matlab_eval':   False,
                       'rpn_file':      None,
                       'min_size':      2
                       }
        assert os.path.exists(self._devkit_path), \
                'VOC devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)

        # for pytorch dataloader
        self._gt_annotations = self.gt_roidb()
        self._is_training = is_training

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        :param i:
        :return:
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Image path does not exist: {}'.format(image_path)
        return image_path


if __name__ == '__main__':
    cfg.data_dir = './data/pascal_voc/'
    d = pascal_voc('trainval', '0712')
    res = d.roidb
    from IPython import embed

    embed()