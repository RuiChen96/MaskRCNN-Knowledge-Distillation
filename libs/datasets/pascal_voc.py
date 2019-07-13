# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torch

import os
from libs.datasets.imdb import imdb
import libs.datasets.ds_utils as ds_utils
from libs.datasets.voc_eval import voc_eval
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

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        :return:
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'ImageSet path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.data_dir, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('Wrote gt roidb to {}',format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
                'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.data_dir,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
                'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            # ? ? ?
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            # filter small boxes
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)
        # end_for
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print('Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs)))
            objs = non_diff_objs
        # end_if
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            #
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            #
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
        # end_for
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas
        }

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        # {:s} is for cls label
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename
        )
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets.size <= 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write(
                            '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1)
                        )

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml'
        )
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt'
        )
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        # use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric
            )
            aps += [ap]
            print('AP for {} = {:.4f}, Recall {:.4f}'.format(cls, ap, rec.max()))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        # end_for
        # each cls has the same weight for the mean value
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir = 'output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')

        # path = os.path.join(cfg.data_dir, 'lib', 'datasets',
        #                     'VOCdevkit-matlab-wrapper')
        # cmd = 'cd {} && '.format(path)
        # cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        # cmd += '-r "dbstop if error; '
        # cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
        #        .format(self._devkit_path, self._get_comp_id(),
        #                self._image_set, output_dir)
        # print('Running:\n{}'.format(cmd))
        # status = subprocess.call(cmd, shell=True)
        pass

    def evaluate_detections(self, all_boxes, output_dir='output'):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    ######################################################
    ### for pytorch dataloader
    ######################################################

    def to_detection_format(self, Dets, image_ids, im_scale_list):
        """
        Add detection results to list
        """
        for i, (dets, img_id) in enumerate(zip(Dets, image_ids)):
            im_scale = im_scale_list[i]
            dets[:, 0:4] = dets[:, 0:4] / im_scale
        return Dets

    def to_evaluation_format(self, all_results):
        """
        Return a list of list num_classes x num_images,
        each element is a numpy.array of N x 5,
        ( [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...] )

        Input: [x1, y1, x2, y2, score, class_id]
        Output: num_classes x num_images list [x1, y1, x2, y2, score]
        """
        all_boxes = [[[] for _ in xrange(self.num_classes)]
                     for _ in xrange(self.num_classes)]

        for img_id in range(self.num_images):
            cls_dets = all_results[img_id]
            cls_ids = cls_dets[:, -1].astype(np.int32)
            for class_id in range(1, self.num_classes):
                inds = np.where(cls_ids == class_id)[0]
                all_boxes[class_id][img_id] = cls_dets[inds, :5]

        return all_boxes

    def __len__(self):
        return len(self._image_index)

    def __getitem__(self, i):
        img_anno = self._gt_annotations[i]
        img_name = self.image_path_at(i)
        bboxes = img_anno['boxes'].astype(np.float32)
        classes = img_anno['gt_classes'].astype(np.int32)
        height, width = cv2.imread(img_name).shape[:2]

        # ignore instance masks and mask training by building zero arrays
        n = classes.shape[0]
        inst_masks = np.zeros([n, height, width], dtype=np.int32)
        mask = np.zeros([height, width], dtype=np.int32)

        im, im_scale, annots = data_layer_keep_aspect_ratio(
            img_name, bboxes, classes, inst_masks, mask, self._is_training
        )

        img_id = os.path.splitext(os.path.basename(img_name))[0]

        return im, im_scale, annots, img_id


def collate_fn(data):
    pass


def collate_fn_testing(data):
    im_batch, im_scale_batch, anchors, _, _, _ = \
        data_layer_keep_aspect_ratio_batch(data, is_training=False)


def get_loader(data_dir, split, is_training, batch_size=16, shuffle=True, num_workers=4):

    # init
    split_, year = split.split('_')
    dataset = pascal_voc(split_, year, is_training=is_training)

    if is_training:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn)
    else:
        return sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn_testing)


if __name__ == '__main__':
    cfg.data_dir = './data/pascal_voc/'
    d = pascal_voc('trainval', '0712')
    res = d.roidb
    from IPython import embed

    embed()
