from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os.path as osp
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
from nets.vgg16 import vgg16
from model.test import _get_blobs
import numpy as np
import cv2

from model.bbox_transform import clip_boxes, bbox_transform_inv

quantBitDepth = 8
maxQuantCoeff = 2**quantBitDepth - 1

rec_feat_dir = '/abs_path/coding/detection/vgg'
quant_feat_dir = '/abs_path/coding/detection/vgg'
# feat_dir = '/home/zchen/worktable/Faster_RCNN_TF/features/vgg16'
result_dir = '/abs_path/coding/detection/vgg_results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

Qp_list = [12, 22, 32, 42]
feat_list = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4', 'pool4', 'conv5']

cfg_from_file('experiments/cfgs/vgg16.yml')
cfg_from_list(['ANCHOR_SCALES','[8,16,32]','ANCHOR_RATIOS','[0.5,1,2]'])

filename = os.path.splitext(os.path.basename('output/vgg16/voc_2007_trainval+voc_2012_trainval/default/vgg16_faster_rcnn_iter_110000.ckpt'))[0]

tag = ''
tag = tag if tag else 'default'
filename = tag + '/' + filename

imdb = get_imdb('voc_2007_test')
imdb.competition_mode(True)

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
sess = tf.Session(config=tfconfig)

net = vgg16()
net.create_architecture("TEST", imdb.num_classes, tag='default',
                        anchor_scales=cfg.ANCHOR_SCALES,
                        anchor_ratios=cfg.ANCHOR_RATIOS)
saver = tf.train.Saver()
saver.restore(sess, 'output/vgg16/voc_2007_trainval+voc_2012_trainval/default/vgg16_faster_rcnn_iter_110000.ckpt')

conv1_feat_t = tf.get_default_graph().get_tensor_by_name("vgg_16/conv1/conv1_2/Relu:0")
pool1_feat_t = tf.get_default_graph().get_tensor_by_name("vgg_16/pool1/MaxPool:0")
conv2_feat_t = tf.get_default_graph().get_tensor_by_name("vgg_16/conv2/conv2_2/Relu:0")
pool2_feat_t = tf.get_default_graph().get_tensor_by_name("vgg_16/pool2/MaxPool:0")
conv3_feat_t = tf.get_default_graph().get_tensor_by_name("vgg_16/conv3/conv3_3/Relu:0")
pool3_feat_t = tf.get_default_graph().get_tensor_by_name("vgg_16/pool3/MaxPool:0")
conv4_feat_t = tf.get_default_graph().get_tensor_by_name("vgg_16/conv4/conv4_3/Relu:0")
pool4_feat_t = tf.get_default_graph().get_tensor_by_name("vgg_16/pool4/MaxPool:0")
conv5_feat_t = tf.get_default_graph().get_tensor_by_name("vgg_16/conv5/conv5_3/Relu:0")

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def proc_result(scores, bbox_pred, rois, img, im_scale):
    boxes = rois[:, 1:5] / im_scale[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, img.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    return scores, pred_boxes

def read_yuv(yuv_path, data_shape):
    with open(yuv_path, 'r') as fp:
        strs = fp.read()
    data = np.frombuffer(strs, dtype=np.uint8)
    data.shape = (data_shape[2], data_shape[0], data_shape[1])
    return np.transpose(data, (1, 2, 0))

def dequant(quant_data, maxLog2Val, pad_size):
    recData = np.exp2(quant_data[:quant_data.shape[0]-pad_size[0], :quant_data.shape[1]-pad_size[1], :].astype(np.float32) / maxQuantCoeff * maxLog2Val) - 1
    return np.expand_dims(recData.astype(np.float32), axis=0)

def proc_dequant_yuv(yuv_path, maxLog2Val, pad_size, yuv_size):
    yuv_data = read_yuv(yuv_path, yuv_size)
    rec_data = dequant(yuv_data, maxLog2Val, pad_size)
    return rec_data

def get_feat(yuv_data_dir, feat_name, Qp_value, meta_data):
    maxLog2Val, pad_size, yuv_size = meta_data
    yuv_dir = os.path.join(yuv_data_dir, 'rec_{}_Qp{}.yuv'.format(feat_name, Qp_value))
    dequant_data = proc_dequant_yuv(yuv_dir, maxLog2Val, pad_size, yuv_size)
    return dequant_data

num_images = len(imdb.image_index)
# subdirs = os.listdir(rec_feat_dir)
# subdirs = sorted(subdirs)
# total_amount = len(subdirs)
for i in range(num_images):
    print('{} / {}'.format(i+1,num_images))
    sys.stdout.flush()
    img_name = imdb.image_path_at(i)
    im = cv2.imread(img_name)
    blobs, im_scales = _get_blobs(im)
    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    feed_dict = {net._image: blobs['data'], net._im_info: blobs['im_info']}
    cls_score, cls_prob, bbox_pred, rois = sess.run([net._predictions["cls_score"],
                                                net._predictions['cls_prob'],
                                                net._predictions['bbox_pred'],
                                                net._predictions['rois']],
                                                feed_dict=feed_dict)
    tmp_scores, tmp_pred_boxes = proc_result(cls_prob, bbox_pred, rois, im, im_scales)

    subdir = os.path.splitext(os.path.split(img_name)[-1])[0]
    save_result_dir = os.path.join(result_dir, subdir)
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    pickle_save(os.path.join(save_result_dir,'original.pkl'), (tmp_scores, tmp_pred_boxes))

    feat_data_dir = os.path.join(rec_feat_dir, subdir, 'output_yuv')
    meta_data_dir = os.path.join(quant_feat_dir, subdir)
    im_info = blobs['im_info']
    for feat_type in feat_list:
        feat_tensor = locals()[feat_type+'_feat_t']
        meta_dir = os.path.join(meta_data_dir, feat_type+'_meta.pkl')
        metadata = pickle_load(meta_dir)
        for Qp in Qp_list:
            # npy_dir = os.path.join(feat_data_dir, '{}_Qp{}.npy'.format(feat_type, Qp))
            # feat = np.load(npy_dir)
            feat = get_feat(feat_data_dir, feat_type, Qp, metadata)
            cls_score, cls_prob, bbox_pred, rois = sess.run([net._predictions["cls_score"],
                                                net._predictions['cls_prob'],
                                                net._predictions['bbox_pred'],
                                                net._predictions['rois']],
                                                feed_dict={feat_tensor: feat, net._im_info: im_info})
            tmp_scores, tmp_pred_boxes = proc_result(cls_prob, bbox_pred, rois, im, im_scales)
            pickle_save(os.path.join(save_result_dir,'{}_Qp{}.pkl'.format(feat_type, Qp)), (tmp_scores, tmp_pred_boxes))
