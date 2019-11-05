import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import pickle
import numpy as np
import sys
import os

from res50_extract import EmbedNetwork
torch.multiprocessing.set_sharing_strategy('file_system')

data_dir = '/abs_path/VehicleID_V1.0'
model_file = '/abs_path/retrieval/model/25000model_trip_soft_res50_v2.pkl'
# image_path = '/abs_path/VehicleID_V1.0/image/'
save_dir = '/abs_path/coding/retrieval/embeddings_resnet'
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

# trans_tuple = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
#     ])
# Lambda = transforms.Lambda(
#     lambda crops: [trans_tuple(crop) for crop in crops])
# trans = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.TenCrop((256, 256)),
#     Lambda,
# ])

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
    # img = Image.open(os.path.join(image_path, ref+'.jpg'))
    # img = trans(img)
    # img = np.expand_dims(img_data_dic[ref], axis=0)
    # im = torch.from_numpy(img)
    im = img_data_dic[ref].unsqueeze(0)
    im = im.cuda()
    embd, _ = model(im)
    embd = embd.detach().cpu().numpy()
    embeddings.append(embd)

embeddings = np.vstack(embeddings)

np.save(os.path.join(save_dir, 'reference_ori.npy'), embeddings)
