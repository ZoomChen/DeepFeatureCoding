from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import pickle

quant_feat_dir = '/abs_path/coding/retrieval/vgg'

Qp_list = [12, 22, 32, 42]
feat_list = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5']

def read_log(file_name):
    total_volume = 0
    with open(file_name, 'r') as f:
        content = f.readlines()
    for str_line in content:
        if str_line.startswith('POC'):
            component_list = [j for j in str_line.split(' ') if j!='']
            bits_index = component_list.index('bits')
            bits_num = int(component_list[bits_index-1])
            total_volume += bits_num
    # add meta data: [pad_h 3 bit, pad_w 3 bit], maxLog2Val 32 bit, [feat_h 12 bit(to 4096) feat_w 12 bit]
    return total_volume + 6 + 32 + 24

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

subdirs = os.listdir(quant_feat_dir)
subdirs = sorted(subdirs)
total_amount = len(subdirs)

for feat_type in feat_list:
    for Qp in Qp_list:
        log_type = 'enc_{}_Qp{}.log'.format(feat_type, Qp)
        comp_rate_list = np.zeros([total_amount,])
        for (idx,subdir) in enumerate(subdirs):
            log_dir = os.path.join(quant_feat_dir, subdir, 'encode_log', log_type)
            compressed_bits = read_log(log_dir)
            _, pad_size, yuv_size, _ = pickle_load(os.path.join(quant_feat_dir, subdir, feat_type+'_meta.pkl'))
            ori_bits = (yuv_size[0]-pad_size[0])*(yuv_size[1]-pad_size[1])*yuv_size[2]*32
            comp_rate_list[idx] = compressed_bits / ori_bits
        print('{} @ {}: mean {}, std {}'.format(feat_type, Qp, np.mean(comp_rate_list), np.std(comp_rate_list)))
        sys.stdout.flush()
