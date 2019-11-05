import os
import sys
import pickle
import numpy as np

from multiprocessing import Queue, Process, Manager, Lock
import multiprocessing

number_threads = 32
quant_feat_dir = '/abs_path/coding/detection/resnet'
encode_app = '/abs_path/HM-16.12/bin/TAppEncoderStatic'
# decode_app = '/home/zchen/worktable/HM-16.12/bin/TAppDecoderStatic'
encoder_cfg = '/abs_path/HM-16.12/bin/encoder_intra_main_rext.cfg'

Qp_list = [12, 22, 32, 42]

feat_list = ['conv1', 'pool1', 'conv2', 'conv3', 'conv4']

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

def exec_encode(op_path, feat_name):
    input_yuv = os.path.join(op_path, feat_name+'.yuv')
    _, _, yuv_size = pickle_load(os.path.join(op_path, feat_name+'_meta.pkl'))

    output_yuv = os.path.join(op_path, 'output_yuv')
    if not os.path.exists(output_yuv):
        os.makedirs(output_yuv)
    output_bin = os.path.join(op_path, 'output_bin')
    if not os.path.exists(output_bin):
        os.makedirs(output_bin)
    log_dir = os.path.join(op_path, 'encode_log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for Qp in Qp_list:
        rec_yuv = os.path.join(output_yuv, 'rec_{}_Qp{}.yuv'.format(feat_name, Qp))
        bitstream = os.path.join(output_bin, '{}_Qp{}.bin'.format(feat_name, Qp))
        log_file = os.path.join(log_dir, 'enc_{}_Qp{}.log'.format(feat_name, Qp))
        cmd = "{} -c {} -i {} -wdt {} -hgt {} -fr 30 --InputChromaFormat=400 -o {} -q {} -f {} -b {} > {} 2>/dev/null".format(
                encode_app, encoder_cfg, input_yuv, yuv_size[1], yuv_size[0], rec_yuv, Qp, yuv_size[2], bitstream, log_file)
        flag = os.system(cmd)
        if flag !=0:
            print('error occured when encode {}/{} at Qp {}.'.format(op_path, feat_name, Qp))

def proc_thread(subdir_list, number_threads, process_id):
    total_amount = len(subdir_list)
    for (idx,subdir) in enumerate(subdir_list):
        if (idx % number_threads == process_id):
            print('Thread {}: {} / {}'.format(process_id,idx+1,total_amount))
            sys.stdout.flush()
            op_dir = os.path.join(quant_feat_dir, subdir)
            for i in feat_list:
                exec_encode(op_dir, i)

if __name__ == "__main__":
    subdirs = os.listdir(quant_feat_dir)
    subdirs = sorted(subdirs)

    jobs_writer = []
    for i in range(number_threads):
        p = Process(target=proc_thread , args=(subdirs, number_threads , i))
        jobs_writer.append(p)
        p.start()

    for job in jobs_writer:
        # print "wait thread " + str(job)
        job.join()