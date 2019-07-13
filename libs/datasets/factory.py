from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pascal_voc import get_loader as pascal_dataloader

import libs.configs.config as cfg


def get_data_loader(datasetname=None):

    if datasetname is None:
        datasetname = cfg.datasetname

    if datasetname == 'pascal_voc':
        return pascal_dataloader
    else:
        raise ValueError('Dataset {} is not supported.'.format(datasetname))
