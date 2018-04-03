import numpy as np
import six
from six import __init__

def loc2bbox():
    pass

def bbox2loc():
    pass

def bbox_iou(bbox_a, bbox_b):

    bbox_a.shape[1] = 4

def generate_anchor_base(base_size = 16, ratios = [0.5, 1, 2],
                         anchor_scales = [8, 16, 32]):

    py = base_size / 2.
    px = base_size / 2.
    # 