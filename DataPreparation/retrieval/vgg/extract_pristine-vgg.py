import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import pickle
import numpy as np
import sys
import os

from vgg_extract import VGGM as EmbedNetwork
torch.multiprocessing.set_sharing_strategy('file_system')

data_dir = '/abs_path/VehicleID_V1.0'
model_file = '/abs_path/retrieval/model/40000_model_trip_VehicleID_v2.pkl'
save_dir = '/abs_path/coding/retrieval/embeddings_vgg'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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
filelist_database = pickle_load(os.path.join(data_dir, 'database_list-800.pkl'))

## restore model
model = EmbedNetwork(num_class = num_class).cuda()
model.load_state_dict(torch.load(model_file))
model = nn.DataParallel(model)
model.eval()

# np.expand_dims(x, axis=0)
embeddings = []
for ind, query in enumerate(filelist_query):
    im = img_data_dic[query].unsqueeze(0)
    im = im.cuda()
    embd, _ = model(im)
    embd = embd.detach().cpu().numpy()
    embeddings.append(embd)

embeddings = np.vstack(embeddings)

np.save(os.path.join(save_dir, 'query_ori.npy'), embeddings)

embeddings = []
for ind, ref in enumerate(filelist_database):
    im = img_data_dic[ref].unsqueeze(0)
    im = im.cuda()
    embd, _ = model(im)
    embd = embd.detach().cpu().numpy()
    embeddings.append(embd)

embeddings = np.vstack(embeddings)

np.save(os.path.join(save_dir, 'reference_ori.npy'), embeddings)
