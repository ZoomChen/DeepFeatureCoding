from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os.path as osp
import sys
if './tools' not in sys.path:
    sys.path.insert(0, './tools')

import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os

import tensorflow as tf
from nets.resnet_v1 import resnetv1
from model.test import _get_blobs
import numpy as np
import cv2

feat_dir = '/abs_path/ori_features/detection/resnet'
if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

cfg_from_file('experiments/cfgs/res101.yml')
cfg_from_list(['ANCHOR_SCALES','[8,16,32]','ANCHOR_RATIOS','[0.5,1,2]'])

filename = os.path.splitext(os.path.basename('output/res101/voc_2007_trainval+voc_2012_trainval/default/res101_faster_rcnn_iter_110000.ckpt'))[0]

tag = ''
tag = tag if tag else 'default'
filename = tag + '/' + filename

imdb = get_imdb('voc_2007_test')
imdb.competition_mode(True)

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
sess = tf.Session(config=tfconfig)

net = resnetv1(num_layers=101)
net.create_architecture("TEST", imdb.num_classes, tag='default',
                        anchor_scales=cfg.ANCHOR_SCALES,
                        anchor_ratios=cfg.ANCHOR_RATIOS)
saver = tf.train.Saver()
saver.restore(sess, 'output/res101/voc_2007_trainval+voc_2012_trainval/default/res101_faster_rcnn_iter_110000.ckpt')

conv1_feat_t = tf.get_default_graph().get_tensor_by_name("resnet_v1_101/conv1/Relu:0")
pool1_feat_t = tf.get_default_graph().get_tensor_by_name("resnet_v1_101/pool1/MaxPool:0")
conv2_feat_t = tf.get_default_graph().get_tensor_by_name("resnet_v1_101_1/block1/unit_3/bottleneck_v1/Relu:0")
conv3_feat_t = tf.get_default_graph().get_tensor_by_name("resnet_v1_101_2/block2/unit_4/bottleneck_v1/Relu:0")
conv4_feat_t = tf.get_default_graph().get_tensor_by_name("resnet_v1_101_2/block3/unit_23/bottleneck_v1/Relu:0")

num_images = len(imdb.image_index)
for i in range(num_images):
    print('{} / {}'.format(i+1,num_images))
    img_name = imdb.image_path_at(i)
    im = cv2.imread(img_name)
    blobs, im_scales = _get_blobs(im)
    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    feed_dict = {net._image: blobs['data'], net._im_info: blobs['im_info']}
    conv1_feat, pool1_feat, conv2_feat, conv3_feat, conv4_feat = sess.run([conv1_feat_t,pool1_feat_t,
                                                    conv2_feat_t,conv3_feat_t,
                                                    conv4_feat_t],
                                                    feed_dict=feed_dict)

    file_prefix = os.path.splitext(os.path.split(img_name)[-1])[0]
    save_dir = os.path.join(feat_dir, file_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'conv1.npy'), conv1_feat)
    np.save(os.path.join(save_dir, 'pool1.npy'), pool1_feat)
    np.save(os.path.join(save_dir, 'conv2.npy'), conv2_feat)
    np.save(os.path.join(save_dir, 'conv3.npy'), conv3_feat)
    np.save(os.path.join(save_dir, 'conv4.npy'), conv4_feat)
    np.save(os.path.join(save_dir, 'im_info.npy'), blobs['im_info'])
    sys.stdout.flush()
