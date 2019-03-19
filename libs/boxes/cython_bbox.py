from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pkg_resources
import imp


def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    __file__ = pkg_resources.resource_filename(__name__, 'cython_bbox.so')
    __loader__ = None
    del __bootstrap__, __loader__
    imp.load_dynamic(__name__, __file__)


if __name__ == '__main__':
    __bootstrap__()

