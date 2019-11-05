import torch
import torch.nn as nn
import torchvision

import pickle
import numpy as np
import sys
import os

from res50_extract import EmbedNetwork
torch.multiprocessing.set_sharing_strategy('file_system')

data_dir = '/abs_path/VehicleID_V1.0'
model_file = '/abs_path/retrieval/model/25000model_trip_soft_res50_v2.pkl'
feat_dir = '/abs_path/ori_features/retrieval/resnet'
if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

num_class = 13164

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

img_data_dic = pickle_load(os.path.join(data_dir, 'img_data_800.pkl'))
filelist_query = pickle_load(os.path.join(data_dir, 'query_list-800.pkl'))

## restore model
model = EmbedNetwork(num_class = num_class).cuda()
model.load_state_dict(torch.load(model_file))
model = nn.DataParallel(model)
model.eval()

num_images = len(filelist_query)
for ind, query in enumerate(filelist_query):
    print('{} / {}'.format(ind+1,num_images))
    sys.stdout.flush()
    im = img_data_dic[query].unsqueeze(0)
    im = im.cuda()
    conv1, pool1, conv2, conv3, conv4, conv5, pool5, embd = model(im)

    save_dir = os.path.join(feat_dir, query)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'conv1.npy'), conv1.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'pool1.npy'), pool1.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'conv2.npy'), conv2.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'conv3.npy'), conv3.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'conv4.npy'), conv4.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'conv5.npy'), conv5.detach().cpu().numpy())
