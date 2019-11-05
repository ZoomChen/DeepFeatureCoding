import pickle
import numpy as np
import sys
import os

embedding_dir = '/abs_path/coding/retrieval/embeddings_vgg'
list_dir = '/abs_path/VehicleID_V1.0'
save_dir = '/abs_path/coding/retrieval/embeddings_vgg/pristine_index_list.pkl'

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)

def pickle_load(fname):
    with open(fname, 'rb') as pf:
        data = pickle.load(pf)
    return data

filelist_query = pickle_load(os.path.join(list_dir, 'query_list-800.pkl'))
filelist_database = pickle_load(os.path.join(list_dir, 'database_list-800.pkl'))
query_ref_dic = pickle_load(os.path.join(list_dir, 'query_ref_dic-800.pkl'))

query_embeddings = np.load(os.path.join(embedding_dir, 'query_ori.npy'))
ref_embeddings = np.load(os.path.join(embedding_dir, 'reference_ori.npy'))

def cos_sim(A, B):
    '''
    A is (512,) ; B is (n, 512)
    '''
    num = np.dot(B, A)
    denom = np.linalg.norm(A) * np.linalg.norm(B, axis=1)
    cos = num / denom
    # sim = 0.5 + 0.5 * cos # to simplify the computation
    return cos

# ind_lists_ori = np.zeros([len(filelist_query), len(filelist_database)], dtype=np.int64)
ind_lists_ori = []
AP_list = []
for ind_q, q in enumerate(filelist_query):
    print('{} / {}'.format(ind_q+1,len(filelist_query)))
    q_feat = query_embeddings[ind_q]
    cos_list = cos_sim(q_feat, ref_embeddings)
    index_arr = np.argsort(cos_list)[::-1] # from big to small
    index_arr = list(index_arr)

    ind_lists_ori.append(index_arr)

    ref_inds = query_ref_dic[q]
    tmp_orders = []
    for i in ref_inds:
        order_num = index_arr.index(i)
        tmp_orders.append(order_num)
    tmp_orders = sorted(tmp_orders)
    score = 0.
    for i, ind in enumerate(tmp_orders):
        score += ((i + 1) / (ind + 1))
    AP_score = score / len(tmp_orders)
    AP_list.append(AP_score)

mAP_score = np.mean(AP_list)
pickle_save(save_dir, ind_lists_ori)
# print(AP_list)
print(mAP_score)
