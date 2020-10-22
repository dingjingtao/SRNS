import math
import tensorflow as tf
import numpy as np
import pdb
import os
import sys
from sklearn import datasets, linear_model, metrics
import random
import pickle
from time import time
from multiprocessing import Pool

def eval_one_user(id):
    score = _score_eval[id, 1:]
    target_score = _score_eval[id, 0]
    rank = 0
    for each in score:
        if each>target_score:
            rank+=1
        if rank>3:
            break
    recall = []
    ndcg = []
    for topk in [1,3]:
        if rank<topk:
            recall.append(1)
            ndcg.append(math.log(2)/math.log(rank+2))
        else:
            recall.append(0)
            ndcg.append(0)
    return recall, ndcg


def eval(model, sess, test_data, test_data_neg):
    global _score_eval
    global _test_data
    global _test_data_neg
    _test_data = test_data
    _test_data_neg = test_data_neg
    score_eval = []    

    for i in range(test_data.shape[0]):
        user_input = np.ones((101,1))*test_data[i,0]
        item_input = np.reshape(np.array([test_data[i,1]]+test_data_neg[i].tolist()),[-1,1])
        feed_dict = {model.user_input:user_input,
                     model.item_input_pos:item_input}
        score_eval.append(np.reshape(sess.run([model.score],feed_dict),[1,-1]))
    score_eval = np.concatenate(score_eval, axis=0)
    _score_eval = score_eval

    pool = Pool(20)
    res = pool.map(eval_one_user, range(test_data.shape[0]))
    pool.close()
    pool.join()

    Recall = []
    NDCG = []
    for i in range(2): 
        Recall.append(np.mean(np.array([r[0][i] for r in res])))
        NDCG.append(np.mean(np.array([r[1][i] for r in res])))
        
    return Recall, NDCG


def predict_fast(model, sess, num_user, num_item, parallel_users, predict_data=None):
    scores = []
    for s in range(int(num_user/parallel_users)):
        user_input = []
        item_input = []
        for i in range(s*parallel_users,(s+1)*parallel_users):
            user_input.append(np.ones((predict_data.shape[1], 1)) * i)
            item_input.append(np.reshape(predict_data[i], [-1, 1]))
        user_input = np.concatenate(user_input,axis=0)
        item_input = np.concatenate(item_input, axis=0)
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input}
        scores.append(np.reshape(sess.run([model.score], feed_dict), [parallel_users, -1]))
    if int(num_user / parallel_users) * parallel_users < num_user:
        user_input = []
        item_input = []
        for i in range(int(num_user/parallel_users)*parallel_users,num_user):
            user_input.append(np.ones((predict_data.shape[1], 1)) * i)
            item_input.append(np.reshape(predict_data[i], [-1, 1]))
        user_input = np.concatenate(user_input, axis=0)
        item_input = np.concatenate(item_input, axis=0)
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input}
        scores.append(np.reshape(sess.run([model.score], feed_dict),
                                 [num_user-int(num_user/parallel_users)*parallel_users, -1]))
    scores = np.concatenate(scores, axis=0)

    return scores

def predict_pos(model, sess, num_user, max_posid, parallel_users, predict_data=None):
    scores = []
    for s in range(int(num_user/parallel_users)):
        user_input = []
        item_input = []
        for i in range(s*parallel_users,(s+1)*parallel_users):
            user_input.append(np.ones((len(predict_data[i]), 1)) * i)
            item_input.append(np.reshape(predict_data[i], [-1, 1]))
        user_input = np.concatenate(user_input,axis=0)
        item_input = np.concatenate(item_input, axis=0)
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input}
        score_flatten = sess.run(model.score, feed_dict)
        score_tmp = np.zeros((parallel_users, max_posid))

        c = 0
        for i in range(s * parallel_users, (s + 1) * parallel_users):
            l = len(predict_data[i])
            score_tmp[i-s*parallel_users,0:l] = \
                np.reshape(score_flatten[c:c+l, 0], [1, -1])
            c += l
        scores.append(score_tmp)

    if int(num_user / parallel_users) * parallel_users < num_user:
        user_input = []
        item_input = []
        for i in range(int(num_user / parallel_users) * parallel_users, num_user):
            user_input.append(np.ones((len(predict_data[i]), 1)) * i)
            item_input.append(np.reshape(predict_data[i], [-1, 1]))
        user_input = np.concatenate(user_input, axis=0)
        item_input = np.concatenate(item_input, axis=0)
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input}
        score_flatten = sess.run(model.score, feed_dict)
        score_tmp = np.zeros((num_user - int(num_user / parallel_users) * parallel_users, max_posid))

        c = 0
        for i in range(int(num_user / parallel_users) * parallel_users, num_user):
            l = len(predict_data[i])
            score_tmp[i - int(num_user / parallel_users) * parallel_users, 0:l] = \
                np.reshape(score_flatten[c:c + l, 0], [1, -1])
            c += l
        scores.append(score_tmp)
    scores = np.concatenate(scores, axis=0)

    return scores





