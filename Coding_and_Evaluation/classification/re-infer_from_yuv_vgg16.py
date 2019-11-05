from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os.path as osp

import time, os, sys
import pickle
import caffe

import numpy as np

rec_feat_dir = '/abs_path/coding/classification/vgg16'
file_list_dir = 'file_list.npy'
ori_logits_dir = '/abs_path/ori_features/classification/vgg16/ori_logits.npy'
result_file = 'result_vgg16_cls_info_loss.txt'
model_file_dir = '/abs_path/VGG/models'

feat_list = ['conv1_2', 'pool1', 'conv2_2', 'pool2', 'conv3_3', 'pool3', 'conv4_3',
             'pool4', 'conv5_3', 'pool5']
coresponding_op = ['relu1_2', 'conv2_1', 'relu2_2', 'conv3_1', 'relu3_3', 'conv4_1', 'relu4_3',
             'conv5_1', 'relu5_3', 'fc6']

MODEL_DEPLOY_FILE = os.path.join(model_file_dir, 'VGG_ILSVRC_16_layers_deploy.prototxt')
MODEL_WEIGHT_FILE = os.path.join(model_file_dir, 'VGG_ILSVRC_16_layers.caffemodel')

Qp_list = [12, 22, 32, 42]
# Qp_list = [0, 12, 22, 32, 42]

quantBitDepth = 8
maxQuantCoeff = 2**quantBitDepth - 1

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224

MODEL_MEAN_VALUE = np.array([103.939, 116.779, 123.68])

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Classifier( \
    model_file=MODEL_DEPLOY_FILE,
    pretrained_file=MODEL_WEIGHT_FILE,
    image_dims=(MODEL_ORIGINAL_INPUT_SIZE[0], MODEL_ORIGINAL_INPUT_SIZE[1]),
    raw_scale=255., # scale befor mean subtraction
    input_scale=None, # scale after mean subtraction
    mean = MODEL_MEAN_VALUE,
    channel_swap = (2, 1, 0))

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

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

def get_feat(yuv_data_dir, feat_name, Qp_value, meta_data, feat_format='NHWC'):
    maxLog2Val, pad_size, yuv_size = meta_data
    yuv_dir = os.path.join(yuv_data_dir, 'rec_{}_Qp{}.yuv'.format(feat_name, Qp_value))
    dequant_data = proc_dequant_yuv(yuv_dir, maxLog2Val, pad_size, yuv_size)
    if feat_format == 'NCHW':
        dequant_data = np.transpose(dequant_data, (0, 3, 1, 2))
    return dequant_data

# creat empty dictionary for fc8 results
fc8_data = {}
for feat_type in feat_list:
    fc8_data[feat_type] = {}
    for Qp in Qp_list:
        fc8_data[feat_type][Qp] = np.zeros([1000,1000],dtype=np.float32)

# re-infer
filelist = np.load(file_list_dir)
num_images = len(filelist)
for n, fname in enumerate(filelist):
    print('{} / {}'.format(n+1,num_images))
    sys.stdout.flush()
    op_dir = os.path.join(rec_feat_dir, fname[1])
    for i, feat_type in enumerate(feat_list):
        meta_dir = os.path.join(op_dir, feat_type+'_meta.pkl')
        metadata = pickle_load(meta_dir)
        for Qp in Qp_list:
            feat_data_dir = os.path.join(op_dir, 'output_yuv')
            feat = get_feat(feat_data_dir, feat_type, Qp, metadata, 'NCHW')
            net.blobs[feat_type].data[...] = feat
            out = net.forward(start=coresponding_op[i], end='prob',blobs=['fc8'])
            fc8_data[feat_type][Qp][n] = out['fc8']

# calculate fidelity
fc8_ori = np.load(ori_logits_dir)
fc8_ori_ind = np.argmax(fc8_ori, axis=1)
f = open(result_file, 'w')
for i, feat_type in enumerate(feat_list):
    for Qp in Qp_list:
        tmp_fc8_ind = np.argmax(fc8_data[feat_type][Qp], axis=1)
        rel_acc = np.sum(tmp_fc8_ind==fc8_ori_ind) / len(fc8_ori_ind)
        output_str = '{} @ QP{}: {}'.format(feat_type, Qp, rel_acc)
        print(output_str)
        f.write(output_str)
        f.write('\n')
f.close()
