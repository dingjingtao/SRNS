import math
import tensorflow as tf
import numpy as np
import pdb
import os
import sys
import random
from time import time
from multiprocessing import Pool

def eval_one_user(id):
    score = _score_one_epoch[id, :]
    hr_res = []
    ndcg_res = []
    if len(_test_data_merge[id])==0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 0, 0], [1, 3, 5]

    for i in range(len(_test_data_merge[id])):
        target_score = _score_one_epoch[id, _test_data_merge[id][i]]
        rank = 0
        for i in range(score.shape[0]):
            if score[i]>target_score and i not in _train_set[id] :
                rank+=1
            if rank>5:
                break
        hr = []
        ndcg = []
        for topk in [1,3,5]:
            if rank<topk:
                hr.append(1)
                ndcg.append(math.log(2)/math.log(rank+2))
            else:
                hr.append(0)
                ndcg.append(0)
        hr_res.append(hr)
        ndcg_res.append(ndcg)
    hr_res = np.array(hr_res)
    ndcg_res = np.array(ndcg_res)
    recall = []
    ndcg = []
    hr = []
    LgtItem_final = []
    K_list = [1,3,5]
    for i in range(3):
        LgtItem_final.append(min(K_list[i],hr_res.shape[0]))
        dcg_max = 0
        for j in range(hr_res.shape[0]):
            if j<K_list[i]:
                dcg_max+=math.log(2)/math.log(j+2)
        recall.append(np.sum(hr_res[:,i])/LgtItem_final[i])
        ndcg.append(np.sum(ndcg_res[:,i]/dcg_max))
    return recall, ndcg, LgtItem_final


def eval(model, sess, train_set, test_data_merge, num_user, num_item):
    global _score_one_epoch
    global _test_data_merge
    global _train_set
    _test_data_merge = test_data_merge
    _train_set = train_set
    score_one_epoch = []

    for i in range(num_user):
        user_input = np.ones((num_item,1))*i
        item_input = np.reshape(np.array(list(range(num_item))),[-1,1])
        feed_dict = {model.user_input:user_input,
                     model.item_input_pos:item_input}
        score_one_epoch.append(np.reshape(sess.run([model.score],feed_dict),[1,-1]))
    score_one_epoch = np.concatenate(score_one_epoch, axis=0)
    _score_one_epoch = score_one_epoch
    
    pool = Pool(10)
    res = pool.map(eval_one_user, range(len(test_data_merge)))
    pool.close()
    pool.join()
   
    RECALL = []
    NDCG = []
    for i in range(3): 
        RECALL.append(np.mean(np.array([r[0][i] for r in res])))
        NDCG.append(np.mean(np.array([r[1][i] for r in res])))
        
    return RECALL, NDCG, score_one_epoch




