from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import pickle

quantBitDepth = 8
maxQuantCoeff = 2**quantBitDepth - 1

quant_feat_dir = '/abs_path/coding/retrieval/vgg'
rec_feat_dir = '/abs_path/coding/retrieval/vgg_rec'
if not os.path.exists(rec_feat_dir):
    os.makedirs(rec_feat_dir)

Qp_list = [12, 22, 32, 42]

feat_list = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5']

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

def dequant(quant_data, maxLog2Val, pad_size, min_value):
    recData = np.exp2(quant_data[:quant_data.shape[0]-pad_size[0], :quant_data.shape[1]-pad_size[1], :].astype(np.float32) / maxQuantCoeff * maxLog2Val) - 1 + min_value
    return np.expand_dims(recData.astype(np.float32), axis=0)

def proc_dequant_yuv(yuv_path, maxLog2Val, pad_size, yuv_size, min_value, feat_format='NHWC'):
    yuv_data = read_yuv(yuv_path, yuv_size)
    rec_data = dequant(yuv_data, maxLog2Val, pad_size, min_value)
    # Transpose from [n, h, w, c] to [n, c, h, w]
    if feat_format == 'NCHW':
        rec_data = np.transpose(rec_data, (0, 3, 1, 2))
    return rec_data

if __name__ == "__main__":
    subdirs = os.listdir(quant_feat_dir)
    subdirs = sorted(subdirs)
    total_amount = len(subdirs)
    for (idx,subdir) in enumerate(subdirs):
        print('{} / {}'.format(idx+1,total_amount))
        sys.stdout.flush()
        read_dir = os.path.join(quant_feat_dir, subdir, 'output_yuv')
        write_dir = os.path.join(rec_feat_dir, subdir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        for feat_type in feat_list:
            meta_dir = os.path.join(quant_feat_dir, subdir, feat_type+'_meta.pkl')
            maxLog2Val, pad_size, yuv_size, min_v = pickle_load(meta_dir)
            for Qp in Qp_list:
                yuv_dir = os.path.join(read_dir, 'rec_{}_Qp{}.yuv'.format(feat_type, Qp))
                npy_dir = os.path.join(write_dir, '{}_Qp{}.npy'.format(feat_type, Qp))
                dequant_data = proc_dequant_yuv(yuv_dir, maxLog2Val, pad_size, yuv_size, min_v, 'NCHW')
                np.save(npy_dir, dequant_data)
