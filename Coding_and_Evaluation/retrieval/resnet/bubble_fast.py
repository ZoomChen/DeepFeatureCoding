import sys, glob, os
import numpy as np
import pickle

list_dir = '/abs_path/VehicleID_V1.0'
embedding_dir = '/abs_path/coding/retrieval/embeddings_resnet'
pristine_index_dir = '/abs_path/coding/retrieval/embeddings_resnet/pristine_index_list.pkl'

Qp_list = [12, 22, 32, 42]
feat_list = ['conv1', 'pool1', 'conv2', 'conv3', 'conv4', 'conv5']

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

query_embeddings_pristine = np.load(os.path.join(embedding_dir, 'query_ori.npy'))
ref_embeddings = np.load(os.path.join(embedding_dir, 'reference_ori.npy'))

ind_lists_ori = pickle_load(pristine_index_dir)

def cos_sim(A, B):
    '''
    A is (512,) ; B is (n, 512)
    '''
    num = np.dot(B, A)
    denom = np.linalg.norm(A) * np.linalg.norm(B, axis=1)
    cos = num / denom
    # sim = 0.5 + 0.5 * cos # to simplify the computation
    return cos

def bubble_index_fast(index_arr_ori, index_arr_new):
    ref_length = len(index_arr_ori)
    assert ref_length==len(index_arr_new)
    index_arr_ori_dic = {}
    for i,j in enumerate(index_arr_ori):
        index_arr_ori_dic[j] = i
    new_order_list = list(range(ref_length))
    for i,j in enumerate(index_arr_new):
        new_order_list[i] = index_arr_ori_dic[j]
    # print new_order_list
    cnt_swap = 0
    cnt_total = ref_length * (ref_length-1) / 2
    for i in range(1,ref_length):
        # print 'step {}'.format(i)
        cur_num = new_order_list[i]
        if cur_num > new_order_list[i-1]:
            continue
        if cur_num < new_order_list[0]:
            new_order_list[:i+1] = [cur_num] + new_order_list[:i]
            cnt_swap += i
            continue
        # dichotomy start
        first = 0
        last = i - 1
        mid = 0
        while first < last:
            mid = (first + last) // 2
            if cur_num > new_order_list[mid+1]:
                first = mid + 1
            elif cur_num > new_order_list[mid]:
                break
            else:
                last = mid
        new_order_list[:i+1] = new_order_list[:mid+1] + [cur_num] + new_order_list[mid+1:i]
        cnt_swap += i - mid - 1
        # print new_order_list
        # print i - mid - 1
    # print cnt_swap
    # print cnt_total
    return 1 - float(cnt_swap)/cnt_total


for feat_type in feat_list:
    for Qp in Qp_list:
        print('{}_Qp{}:'.format(feat_type, Qp))
        retored_npy_path = os.path.join(embedding_dir, 'query_{}_Qp{}.npy'.format(feat_type, Qp))
        retsored_query_embd = np.load(retored_npy_path)

        bubble_sim_list = []
        AP_list = []
        for ind_q, q in enumerate(filelist_query):
            # print('{}_Qp{}: {} / {}'.format(feat_type, Qp, ind_q+1, len(filelist_query)))
            q_feat = retsored_query_embd[ind_q]
            cos_list = cos_sim(q_feat, ref_embeddings)
            index_arr = np.argsort(cos_list)[::-1] # from big to small
            index_arr = list(index_arr)

            sim_q = bubble_index_fast(ind_lists_ori[ind_q],index_arr)
            bubble_sim_list.append(sim_q)

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
        mean_bubble = np.mean(bubble_sim_list)
        mAP_score = np.mean(AP_list)
        print('  mean bubble similarity: {}'.format(mean_bubble))
        print('  mean average percision: {}'.format(mAP_score))
        sys.stdout.flush()
