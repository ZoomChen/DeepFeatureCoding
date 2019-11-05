from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import time, os, sys
import pickle
import numpy as np

res_tmp_file_dir = '/abs_path/coding/detection/vgg_results_MAP_tmp'

Qp_list = [12, 22, 32, 42]
feat_list = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4', 'pool4', 'conv5']

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

def cal_overlap(BBGT, bb):
    # intersection
    ixmin = np.maximum(BBGT[0], bb[0])
    iymin = np.maximum(BBGT[1], bb[1])
    ixmax = np.minimum(BBGT[2], bb[2])
    iymax = np.minimum(BBGT[3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (BBGT[2] - BBGT[0] + 1.) *
           (BBGT[3] - BBGT[1] + 1.) - inters)

    return inters / uni

def comp_single_cls_img(ori_shots, comp_shots):
    overlap_list = []
    rel_conf_list = []
    weight_list = []
    for i in range(len(ori_shots)):
        max_overlap = 0.
        max_id = None
        for j in range(len(comp_shots)):
            ori_bb = ori_shots[i][:4]
            comp_bb = comp_shots[j][:4]
            ol_rate = cal_overlap(ori_bb, comp_bb)
            if ol_rate > max_overlap:
                max_overlap = ol_rate
                max_id = j
        ori_conf = ori_shots[i][-1]
        if max_id == None:
            overlap_list.append(0.)
            rel_conf_list.append(0.)
            weight_list.append(ori_conf)
        else:
            comp_conf = comp_shots[max_id][-1]
            overlap_list.append(max_overlap)
            rel_conf_list.append(np.maximum(0., (1 - (ori_conf - comp_conf) / ori_conf)))
            weight_list.append(ori_conf)
            comp_shots = np.delete(comp_shots, max_id, 0)
    return overlap_list, rel_conf_list, weight_list

def proc_score(ol, rc, w):
    ol = np.array(ol)
    rc = np.array(rc)
    w = np.array(w)
    return np.sum(ol * rc * w) / np.sum(w)

def compare_dets(ori_list, comp_list):
    score_list = np.zeros((4952,))
    # num_images
    for i in range(4952):
        overlap_rate = []
        rel_conf = []
        weight = []
        # num_classes
        for j in range(1, 21):
            tmp_ori = ori_list[j][i]
            tmp_comp = comp_list[j][i]
            tmp_overlap, tmp_rel_conf, tmp_weight = comp_single_cls_img(tmp_ori, tmp_comp)
            overlap_rate += tmp_overlap
            rel_conf += tmp_rel_conf
            weight += tmp_weight
        score_list[i] = proc_score(overlap_rate, rel_conf, weight)
    return np.mean(score_list)

ori_boxes = pickle_load(os.path.join(res_tmp_file_dir, 'original', 'detections.pkl'))

for feat_type in feat_list:
    for Qp in Qp_list:
        pkl_file = os.path.join(res_tmp_file_dir, '{}_Qp{}'.format(feat_type, Qp), 'detections.pkl')
        tmp_boxes = pickle_load(pkl_file)
        score = compare_dets(ori_boxes, tmp_boxes)
        print('{} @ Qp{}: {}'.format(feat_type, Qp, score))
        sys.stdout.flush()