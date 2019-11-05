from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import pickle

feat_dir = '/abs_path/ori_features/retrieval/resnet'
quant_feat_dir = '/abs_path/coding/retrieval/resnet'
if not os.path.exists(quant_feat_dir):
    os.makedirs(quant_feat_dir)

feat_list = ['conv1', 'pool1', 'conv2', 'conv3', 'conv4', 'conv5']

quantBitDepth = 8
maxQuantCoeff = 2**quantBitDepth - 1

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

def quant(feat, feat_format='NHWC'):
    # pytorch tensor shape: [n, channel, h, w]
    if feat_format == 'NCHW':
        feat = np.transpose(feat, (0, 2, 3, 1))
    pad_size_h = np.mod(feat.shape[1], 8)
    if pad_size_h != 0:
        pad_size_h = 8 - pad_size_h
    pad_size_w = np.mod(feat.shape[2], 8)
    if pad_size_w != 0:
        pad_size_w = 8 - pad_size_w
    data_log2 = np.log2(feat+1)
    maxLog2Val = data_log2.max()
    datalog2SampleScaled = np.rint(data_log2 * maxQuantCoeff /  maxLog2Val).astype(np.uint8)
    padded_matrix = np.pad(datalog2SampleScaled, ((0,0),(0, pad_size_h),(0,pad_size_w),(0,0)), 'edge')
    meta_data = (maxLog2Val, (pad_size_h, pad_size_w), (feat.shape[1]+pad_size_h, feat.shape[2]+pad_size_w, feat.shape[3]))
    return padded_matrix, meta_data

def save_yuv(quant_data, save_dir):
    fp = open(save_dir,'w')
    for i in range(quant_data.shape[-1]):
        strs = np.squeeze(quant_data[0,:,:,i]).tobytes()
        print(strs,file=fp,end='')
    fp.close()

subdirs = os.listdir(feat_dir)
subdirs = sorted(subdirs)
total_amount = len(subdirs)
# img_amount = len(subdirs)
for ind,i in enumerate(subdirs):
    print('{} / {}'.format(ind+1,total_amount))
    sys.stdout.flush()
    subdir = os.path.join(feat_dir, i)
    for j in feat_list:
        feat_file = os.path.join(subdir, j+'.npy')
        feat_data = np.load(feat_file)
        # pytorch tensor shape: [n, channel, h, w]
        quant_feat, feat_meta = quant(feat_data, feat_format='NCHW')
        # save files
        tmp_dir = os.path.join(quant_feat_dir, i)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        save_yuv(quant_feat, os.path.join(tmp_dir,j+'.yuv'))
        pickle_save(os.path.join(tmp_dir,j+'_meta.pkl'), feat_meta)
