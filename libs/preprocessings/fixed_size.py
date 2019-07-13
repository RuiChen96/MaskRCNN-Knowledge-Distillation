import cv2
import numpy as np
from libs.boxes.bbox_transform import clip_boxes


def random_flip(im, inst_masks, mask, boxes, classes):
    h, w, c = im.shape
    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)
        mask = cv2.flip(mask, 1)
    if inst_masks.size > 0:
        # inst_masks of shape (n, h, w)
        inst_masks = np.transpose(inst_masks, (1, 2, 0))    # to (h, w, n)
        if flip:
            inst_masks = cv2.flip(inst_masks, 1)
        try:
            if inst_masks.ndim > 2:
                inst_masks = np.transpose(inst_masks, (2, 0, 1))    # to (n, h, w)
            else:
                inst_masks = inst_masks.reshape((1, h, w))
        except ValueError:
            print(inst_masks.ndim, inst_masks.shape)
            raise
    else:
        inst_masks = np.zeros((0, h, w), inst_masks.dtype)

    boxes = _offset_boxes(boxes, im.shape, 1, [0, 0], flip)
    return im, inst_masks, mask, boxes, classes


def resize_as_min_side(im, masks, mask, boxes, classes, min_side, max_side):
    """
    resize image so that it may max-fix the canvas
    """
    h, w, c = im.shape
    n = classes.size
    im_shape = im.shape
    # get the min, max of h, w
    im_size_min = np.min(im_shape[0: 2])
    im_size_max = np.max(im_shape[0: 2])
    im_scale = float(min_side) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_side:
        im_scale = float(max_side) / float(im_size_max)

    new_w, new_h = int(im_scale * w), int(im_scale * h)
    im = cv2.resize(im, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    boxes *= im_scale
    if masks.size > 0:
        # masks of shape (n, h, w)
        masks = np.transpose(masks, (1, 2, 0))  # to (h, w, n)
        masks = cv2.resize(masks, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if masks.ndim > 2:
            masks = np.transpose(masks, (2, 0, 1))  # to (n, h, w)
        else:
            masks = masks.reshape((1, new_h, new_w))

    return im, masks, mask, boxes, classes, im_scale


def _offset_boxes(boxes, im_shape, scale, offs, flip):
    if len(boxes) == 0:
        return boxes

    boxes = np.asarray(boxes, dtype=np.float)
    boxes *= scale
    boxes[:, 0::2] -= offs[0]
    boxes[:, 1::2] -= offs[1]
    boxes = clip_boxes(boxes, im_shape)

    if flip:
        boxes_x = np.copy(boxes[:, 0])
        boxes[:, 0] = im_shape[1] - boxes[:, 2]
        boxes[:, 2] = im_shape[1] - boxes_x

    return boxes


def preprocess_train_keep_aspect_ratio(im, boxes, classes, inst_masks, mask,
                                       min_side, max_side,
                                       canvas_height, canvas_width,
                                       use_augment=False, training_scale=[0.3, 0.5, 0.7, 1.0]):
    """
    Pre-processing images, boxes, classes, etc.
    :param im: of shape (H, W, 3)
    :param boxes: of shape (N, 4)
    :param classes: of shape (N, )
    :param inst_masks: of shape (N, H, W)
    :param mask: of shape (H, W)
    input_size: a list of specified input shapes (ih, iw)

    :return:
    """
    im, inst_masks, mask, boxes, classes, im_scale = resize_as_min_side(im, inst_masks, mask, boxes, classes,
                                                                        min_side=min_side, max_side=max_side)
    im, inst_masks, mask, boxes, classes = random_flip()
