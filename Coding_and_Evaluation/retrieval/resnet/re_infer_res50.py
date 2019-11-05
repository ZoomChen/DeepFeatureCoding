import torch
import torch.nn as nn
import torchvision

import pickle
import numpy as np
import sys
import os

from res50_ret_eval import EmbedNetwork
torch.multiprocessing.set_sharing_strategy('file_system')

model_file = '/abs_path/retrieval/model/25000model_trip_soft_res50_v2.pkl'
rec_feat_dir = '/abs_path/coding/retrieval/resnet_rec'
list_dir = '/abs_path/VehicleID_V1.0'
result_dir = '/abs_path/coding/retrieval/embeddings_resnet'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

Qp_list = [12, 22, 32, 42]
feat_list = ['conv1', 'pool1', 'conv2', 'conv3', 'conv4', 'conv5']

num_class = 13164

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

filelist_query = pickle_load(os.path.join(list_dir, 'query_list-800.pkl'))

model = EmbedNetwork(pretrained_base = False, num_class = num_class).cuda()
model.load_state_dict(torch.load(model_file))
model = nn.DataParallel(model)
model.eval()

for feat_type in feat_list:
    for Qp in Qp_list:
        embeddings = []
        for query in filelist_query:
            npy_dir = os.path.join(rec_feat_dir, query, '{}_Qp{}.npy'.format(feat_type, Qp))
            feat = torch.from_numpy(np.load(npy_dir))
            feat = feat.cuda()
            embd = model(feat, feat_type)
            embd = embd.detach().cpu().numpy()
            embeddings.append(embd)
        embeddings = np.vstack(embeddings)
        np.save(os.path.join(result_dir, 'query_{}_Qp{}.npy'.format(feat_type, Qp)), embeddings)
        print('{}_Qp{} done'.format(feat_type, Qp))
        sys.stdout.flush()
