import numpy as np
import os

feat_dir = '/abs_path/ori_features/classification/vgg16'
save_dir = '/abs_path/ori_features/classification/vgg16'
logit_name = 'fc8' #for vgg16
# logit_name = 'fc1000' #for resnet

filelist = np.load('file_list.npy')

fc8_data = np.zeros((1000, 1000), dtype=np.float32)
for n, fname in enumerate(filelist):
    fc8_data[n] = np.load(os.path.join(feat_dir, fname[1], logit_name+'.npy'))

np.save(os.path.join(save_dir, 'ori_logits.npy'), fc8_data)
