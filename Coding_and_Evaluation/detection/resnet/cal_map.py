from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if './tools' not in sys.path:
    sys.path.insert(0, './tools')

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import pickle

import tensorflow as tf
from nets.resnet_v1 import resnetv1
import numpy as np
from model.nms_wrapper import nms

result_dir = '/abs_path/coding/detection/resnet_results'
res_tmp_file_dir = '/abs_path/coding/detection/resnet_results_MAP_tmp'
if not os.path.exists(res_tmp_file_dir):
    os.makedirs(res_tmp_file_dir)

Qp_list = [12, 22, 32, 42]
feat_list = ['conv1', 'pool1', 'conv2', 'conv3', 'conv4']

cfg_from_file('experiments/cfgs/res101.yml')
cfg_from_list(['ANCHOR_SCALES','[8,16,32]','ANCHOR_RATIOS','[0.5,1,2]'])

imdb = get_imdb('voc_2007_test')
imdb.competition_mode(True)

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

namelist = ['original.pkl']
for feat_type in feat_list:
    for Qp in Qp_list:
        namelist.append('{}_Qp{}.pkl'.format(feat_type, Qp))

max_per_image=100
thresh=0.
num_images = len(imdb.image_index)
for res_file in namelist:
    res_name = os.path.splitext(res_file)[0]
    output_dir = os.path.join(res_tmp_file_dir, res_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
    for i in range(num_images):
        # print('{} -- {} / {}'.format(res_name, i+1,num_images))
        img_name = imdb.image_path_at(i)
        subdir = os.path.splitext(os.path.split(img_name)[-1])[0]
        res_dir = os.path.join(result_dir, subdir, res_file)
        scores, boxes = pickle_load(res_dir)
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('!!! {} !!!'.format(res_name))
    imdb.evaluate_detections(all_boxes, output_dir)
    sys.stdout.flush()
