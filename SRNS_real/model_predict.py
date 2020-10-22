import os
import tensorflow as tf
import numpy as np
from collections import defaultdict
import pdb
import sys
import random
import logging
from Model import MODEL
import argparse
import BatchGenUser as BatchUser
import EvaluateUser as EvalUser
from time import time
from multiprocessing import Pool
import setproctitle
import pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="SRNS")
    parser.add_argument('--process_name', nargs='?', default='SRNS',
                        help='Input process name.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1000,
                        help='batch_size')
    parser.add_argument('--regs', nargs='?', default='0.0',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--embedding_size', type=int, default=32,
                        help='embedding_size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--gpu', type=str, default='3',
                        help='GPU.')
    parser.add_argument('--optimizer', nargs='?', default='Adam',
                        help='Choose an optimizer: GradientDescent, Adagrad, Adam')
    parser.add_argument('--use_pretrain', action='store_true', default=False,
                        help='use pretrain model or not')
    parser.add_argument('--model_file', type=str, default="srns",
                        help='model file path')
    return parser.parse_args()



def training(model, args, train_data, val_data, test_data, test_data_neg):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        Recall, NDCG = EvalUser.eval(model, sess, test_data, test_data_neg)
        print("%s model predict, test data, Recall = %.4f/%.4f, NDCG = %.4f/%.4f" % (args.model_file.upper(), Recall[0], Recall[1], NDCG[0], NDCG[1]))


       

   


def run():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    setproctitle.setproctitle(args.process_name)

    train_data = pickle.load(open('./ml1m/train.pkl', 'rb'))
    val_data = pickle.load(open('./ml1m/val.pkl', 'rb'))
    test_data = pickle.load(open('./ml1m/test.pkl', 'rb'))
    test_data_neg = pickle.load(open('./ml1m/test_neg.pkl', 'rb'))

    num_user = max(np.max(train_data[:, 0]), np.max(test_data[:, 0])) + 1
    num_item = max(np.max(train_data[:, 1]), np.max(test_data[:, 1])) + 1

    graph = tf.Graph()
    with graph.as_default():
        model = MODEL(args, num_user, num_item)
        model.build_graph()
        training(model, args, train_data, val_data, test_data, test_data_neg)


if __name__ == '__main__':
    run()
