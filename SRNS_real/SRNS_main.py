import os
import tensorflow as tf
import numpy as np
from collections import defaultdict
import pdb
import sys
from tqdm import tqdm
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
    parser = argparse.ArgumentParser(description="Run Sampler-GAN.")
    parser.add_argument('--process_name', nargs='?', default='SRNS',
                        help='Input process name.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1000,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs.')
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
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for softmax')
    parser.add_argument('--trial_id', nargs='?', default='0',
                        help='Indicate trail id with same condition')
    parser.add_argument('--early_stop', nargs='?', type=int, default=10,
                        help='early_stop evals')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='save the best model')
    parser.add_argument('--use_pretrain', action='store_true', default=False,
                        help='use pretrain model or not')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha')
    parser.add_argument('--warmup', type=float, default=1.0,
                        help='warmup')
    parser.add_argument('--S1', type=int, default=20,
                        help='size of cache for final sample')
    parser.add_argument('--varset_size', type=int, default=3000,
                        help='size of candidate for var monitor')
    parser.add_argument('--fix_seed', action='store_true', default=False,
                        help='fix random seed')
    parser.add_argument('--model_file', type=str, default="model/model.pkl",
                        help='model file path')
    parser.add_argument('--S2_div_S1', type=int, default=1,
                        help='cache size of cache than cache1')
    return parser.parse_args()



def training(model, args, train_data, val_data, test_data, test_data_neg, num_user, num_item):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        logging.info("--- Start training ---%s" % filename)
        print("--- Start training ---%s" % filename)
        sess.run(tf.global_variables_initializer())

        # init the train_set, test_set
        train_set = defaultdict(set)
        for i in range(train_data.shape[0]):
            train_set[train_data[i, 0]].add(train_data[i, 1])
        test_set = defaultdict(set)
        for i in range(test_data.shape[0]):
            test_set[test_data[i, 0]].add(test_data[i, 1])

        # init the train_iddict [u] pos->id
        train_iddict = [defaultdict(int) for _ in range(num_user)]
        train_pos = [[] for _ in range(num_user)]
        max_posid = 0
        for i in range(num_user):
            poscnt = 0
            max_posid = max(max_posid, len(train_set[i]))
            for p in train_set[i]:
                train_iddict[i][p] = poscnt
                poscnt += 1
                train_pos[i].append(p)
        print("MAX POS IDX: %d"%max_posid)

        # init the two candidate sets for monitoring variance
        candidate_cur = np.random.choice(num_item, [num_user, args.varset_size])
        for i in range(num_user):
            for j in range(args.varset_size):
                while candidate_cur[i, j] in train_set[i]:
                    candidate_cur[i, j] = random.randint(0, num_item - 1)

        candidate_nxt = [np.random.choice(num_item, [num_user, args.varset_size]) for _ in range(5)]
        for c in range(5):
            for i in range(num_user):
                for j in range(args.varset_size):
                    while candidate_nxt[c][i, j] in train_set[i]:
                        candidate_nxt[c][i, j] = random.randint(0, num_item - 1)

        Mu_idx = []  # All possible items or non-fn items
        for i in range(num_user):
            Mu_idx_tmp = random.sample(list(range(args.varset_size)), args.S1)
            Mu_idx.append(Mu_idx_tmp)

        Recall, NDCG = EvalUser.eval(model, sess, val_data, test_data_neg)
        print("Before trianing, val data, Recall = %.4f/%.4f, NDCG = %.4f/%.4f" % (Recall[0], Recall[1], NDCG[0], NDCG[1]))
        logging.info("Before trianing, val data, Recall = %.4f/%.4f, NDCG = %.4f/%.4f" % (Recall[0], Recall[1], NDCG[0], NDCG[1]))

        score_cand_cur = np.array(
            [EvalUser.predict_fast(model, sess, num_user, num_item, parallel_users=100, predict_data=candidate_cur)])
        score_cand_nxt = [np.zeros((0, num_user, args.varset_size)) for _ in range(5)]
        score_pos_cur = np.array(
            [EvalUser.predict_pos(model, sess, num_user, max_posid, parallel_users=100, predict_data=train_pos)])

        Metric_best = 0
        stop_counter = 0

        for epoch_count in range(args.epochs):
            train_begin = time()
            batches = BatchUser.sampling(args, train_data)
            loss, Mu_idx = training_batch(model, sess, batches, args, epoch_count, train_data, num_item,
                                        score_cand_cur, score_pos_cur, Mu_idx, candidate_cur, train_iddict)
            train_time = time() - train_begin

            valid_begin = time()
            Recall, NDCG = EvalUser.eval(model, sess, val_data, test_data_neg)
            valid_time = time() - valid_begin
            print("Epoch %d [%.1fs]: loss=%.4f, Recall = %.4f/%.4f, NDCG = %.4f/%.4f [%.1fs]" % (
            epoch_count + 1, train_time, loss, Recall[0], Recall[1], NDCG[0], NDCG[1], valid_time))
            logging.info("Epoch %d [%.1fs]: loss=%.4f, Recall = %.4f/%.4f, NDCG = %.4f/%.4f [%.1fs]" % (
            epoch_count + 1, train_time, loss, Recall[0], Recall[1], NDCG[0], NDCG[1], valid_time))

            score_1epoch_nxt = []
            for c in range(5):
                score_1epoch_nxt.append(np.array(
                    [EvalUser.predict_fast(model, sess, num_user, num_item, parallel_users=100,
                                            predict_data=candidate_nxt[c])]))
            score_1epoch_pos = np.array(
                [EvalUser.predict_pos(model, sess, num_user, max_posid, parallel_users=100, predict_data=train_pos)])

            # delete the score_cand_cur[0,:,:] at the earlist timestamp
            if epoch_count >= 5 or epoch_count == 0:
                score_pos_cur = np.delete(score_pos_cur, 0, 0)

            for c in range(5):
                score_cand_nxt[c] = np.concatenate([score_cand_nxt[c], score_1epoch_nxt[c]], axis=0)
            score_pos_cur = np.concatenate([score_pos_cur, score_1epoch_pos], axis=0)

            score_cand_cur = np.copy(score_cand_nxt[0])
            candidate_cur = np.copy(candidate_nxt[0])
            for c in range(4):
                candidate_nxt[c] = np.copy(candidate_nxt[c + 1])
                score_cand_nxt[c] = np.copy(score_cand_nxt[c + 1])
            candidate_nxt[4] = np.random.choice(num_item, [num_user, args.varset_size])
            for i in range(num_user):
                for j in range(args.varset_size):
                    while candidate_nxt[4][i, j] in train_set[i]:
                        candidate_nxt[4][i, j] = random.randint(0, num_item - 1)
            score_cand_nxt[4] = np.delete(score_cand_nxt[4], list(range(5)), 0)

            if Recall[0] > Metric_best:
                Model_byR1_param = sess.run([model.embeddingmap_user, model.embeddingmap_item, model.h])
                Metric_best = Recall[0]
                if args.save_model:
                    if not os.path.exists("Model-Final"):
                        os.makedirs("Model-Final")
                    save_path = "Model-Final/model-byR1-%s.pkl"%filename[4:]
                    model.save_model_withpath(sess, save_path)
                    print("Model-by-R1 Saved")
            else:
                stop_counter += 1
                if stop_counter > args.early_stop:
                    print("early stopped")
                    logging.info("early stopped")
                    break

        # Test via Model_byR1_param
        sess.run(tf.assign(model.embeddingmap_user, Model_byR1_param[0]))
        sess.run(tf.assign(model.embeddingmap_item, Model_byR1_param[1]))
        sess.run(tf.assign(model.h, Model_byR1_param[2]))
        Recall, NDCG = EvalUser.eval(model, sess, test_data, test_data_neg)
        print("Test data via Model_byR1_param: Recall = %.4f/%.4f, NDCG = %.4f/%.4f" % (Recall[0], Recall[1], NDCG[0], NDCG[1]))
        logging.info("Test data via Model_byR1_param: Recall = %.4f/%.4f, NDCG = %.4f/%.4f" % (Recall[0], Recall[1], NDCG[0], NDCG[1]))

        return Metric_best


def training_batch(model, sess, batches, args, epoch_cur, train_data, num_item,
                       score_cand_all, score_pos_all, Mu_idx, candidate_cur, train_iddict):
    user_batch, item_batch = batches
    num_batch = len(user_batch)
    loss_average = 0.0

    for batch_idx in tqdm(range(num_batch)):
        negitems=[]
        negitems_candidates_all = []
        for i in range(len(user_batch[batch_idx])):
            negitems_candidates_all.append(Mu_idx[user_batch[batch_idx][i]])
        negitems_candidates_all = np.array(negitems_candidates_all)

        feed_dict = {
            model.user_input: np.reshape(np.array(user_batch[batch_idx]), [-1, 1]),
            model.item_input_pos: np.reshape(item_batch[batch_idx], [-1, 1])
        }
        ratings_positems = sess.run(model.score, feed_dict)
        ratings_positems = np.reshape(ratings_positems, [-1])

        # Sampling from the cache
        Mu_items_all = []
        for i in range(len(user_batch[batch_idx])):
            Mu_items_all.append(candidate_cur[user_batch[batch_idx][i], negitems_candidates_all[i]])
        Mu_items_all = np.reshape(np.array(Mu_items_all), [-1, 1])
        
        users = np.reshape(np.tile(np.reshape(np.array(user_batch[batch_idx]), [-1, 1]),
                                    (1, args.S1)), [-1, 1])
        feed_dict = {
            model.user_input: users,
            model.item_input_neg: Mu_items_all
        }
        ratings_candidates_all = np.reshape(sess.run(model.score_neg, feed_dict), [-1, args.S1])

        hisscore_candidates_all = []
        hisscore_pos_all = []
        for i in range(len(user_batch[batch_idx])):
            user = user_batch[batch_idx][i]
            hisscore_candidates_all.append(
                score_cand_all[:, user:user+1, np.reshape(negitems_candidates_all[i], [-1])]) # 5 * 1 * N
            pos = item_batch[batch_idx][i]
            posid = train_iddict[user][pos]
            hisscore_pos_all.append(score_pos_all[:, user:user+1, posid]) # 5 * 1
        hisscore_candidates_all = np.concatenate(hisscore_candidates_all, axis=1) # 5 * B * N
        hisscore_pos_all = np.expand_dims(np.concatenate(hisscore_pos_all, axis=1), -1) # 5 * B * 1

        hislikelihood_candidates_all = 1 / (1 + np.exp(hisscore_pos_all - hisscore_candidates_all))

        mean_candidates_all = np.mean(hislikelihood_candidates_all[:, :, :], axis=0)
        variance_candidates_all = np.zeros(mean_candidates_all.shape)
        for i in range(hislikelihood_candidates_all.shape[0]):
            variance_candidates_all += (hislikelihood_candidates_all[i, :, :] - mean_candidates_all) ** 2
        variance_candidates_all = np.sqrt(variance_candidates_all / hislikelihood_candidates_all.shape[0])

        likelihood_candidates_all = \
            1 / (1 + np.exp(np.expand_dims(ratings_positems, -1) - ratings_candidates_all))
        
        # Top sampling strategy by score + alpha * std
        if args.alpha >= 0:
            item_arg_all = np.argmax(likelihood_candidates_all +
                                        args.alpha * min(1, epoch_cur/args.warmup)
                                        * variance_candidates_all, axis=1)
        else:
            item_arg_all = np.argmax(variance_candidates_all, axis=1)
        example_weight = np.ones((len(user_batch[batch_idx]),1), dtype=np.float)

        for i in range(len(user_batch[batch_idx])):
            negitems.append(candidate_cur[user_batch[batch_idx][i], negitems_candidates_all[i, item_arg_all[i]]])
  
        # update Mu
        negitems_mu_candidates = []
        for i in range(len(user_batch[batch_idx])):
            Mu_set = set(Mu_idx[user_batch[batch_idx][i]])
            while len(Mu_idx[user_batch[batch_idx][i]]) < args.S1 * (1 + args.S2_div_S1):
                random_item = random.randint(0, candidate_cur.shape[1] - 1)
                while random_item in Mu_set:
                    random_item = random.randint(0, candidate_cur.shape[1] - 1)
                Mu_idx[user_batch[batch_idx][i]].append(random_item)
            negitems_mu_candidates.append(Mu_idx[user_batch[batch_idx][i]])
        negitems_mu_candidates = np.array(negitems_mu_candidates)

        negitems_mu = []
        for i in range(len(user_batch[batch_idx])):
            negitems_mu.append(candidate_cur[user_batch[batch_idx][i], negitems_mu_candidates[i]])
        negitems_mu = np.reshape(np.array(negitems_mu), [-1, 1])
        users = np.reshape(np.tile(np.reshape(np.array(user_batch[batch_idx]), [-1, 1]),
                                    (1, args.S1 * (1 + args.S2_div_S1))), [-1, 1])
        feed_dict = {
            model.user_input: users,
            model.item_input_neg: negitems_mu
        }
        ratings_mu_candidates = np.reshape(sess.run(model.score_neg, feed_dict),
                                            [-1, args.S1 * (1 + args.S2_div_S1)])
        ratings_mu_candidates = ratings_mu_candidates / args.temperature
        ratings_mu_candidates = np.exp(ratings_mu_candidates) / np.reshape(
            np.sum(np.exp(ratings_mu_candidates), axis=1), [-1, 1])

        user_set = set()
        for i in range(len(user_batch[batch_idx])):
            if user_batch[batch_idx][i] in user_set:
                continue
            else:
                user_set.add(user_batch[batch_idx][i])
            cache_arg = np.random.choice(args.S1 * (1 + args.S2_div_S1), args.S1,
                                            p=ratings_mu_candidates[i], replace=False)
            Mu_idx[user_batch[batch_idx][i]] = np.array(Mu_idx[user_batch[batch_idx][i]])[cache_arg].tolist()

        users = np.reshape(user_batch[batch_idx], [-1, 1])
        positems = np.reshape(item_batch[batch_idx], [-1, 1])
        negitems = np.reshape(np.array(negitems), [-1, 1])

        feed_dict = {model.user_input: users,
                        model.item_input_pos: positems,
                        model.item_input_neg: negitems,
                        model.example_weight: example_weight}
        _, loss = sess.run([model.optimizer, model.loss], feed_dict)
        loss_average += loss

    loss_average /= train_data.shape[0]

    return loss_average, Mu_idx


def init_logging_and_result(args):
    Log_dir_name = 'Log'
    if not os.path.exists(Log_dir_name):
        os.makedirs(Log_dir_name)
    F_trail_id = args.trial_id
    F_optimizer = args.optimizer
    F_lr = str(args.lr)
    F_reg = args.regs
    F_alpha = str(args.alpha) + '-' + str(args.warmup)
    F_temp = str(args.temperature)
    F_Mu_size = str(args.varset_size) + '-' + str(args.S1)
    global filename
    filename = "log-SRNS-%s-F%s-lr%s-reg%s-tem%s-alpha%s-S1-%s-S2_div_S1_%d-%s" % (
        F_optimizer, str(args.embedding_size), F_lr, F_reg, F_temp,
        F_alpha, F_Mu_size, args.S2_div_S1, F_trail_id)
    if args.use_pretrain:
        filename += "_use_pretrain"
    if not os.path.exists(Log_dir_name + '/' + filename):
        logging.basicConfig(filename=Log_dir_name + '/' + filename, level=logging.INFO)
    else:
        print(Log_dir_name + '/' + filename, 'already exists, skipping ...')
        exit(0)


def run():
    args = parse_args()
    if args.fix_seed:
        random.seed(1)
        np.random.seed(1)
        print("Fix Random Seed!!!")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    setproctitle.setproctitle(args.process_name)

    train_data = pickle.load(open('./ml1m/train.pkl', 'rb'))
    val_data = pickle.load(open('./ml1m/val.pkl', 'rb'))
    test_data = pickle.load(open('./ml1m/test.pkl', 'rb'))
    test_data_neg = pickle.load(open('./ml1m/test_neg.pkl', 'rb'))

    num_user = max(np.max(train_data[:, 0]), np.max(test_data[:, 0])) + 1
    num_item = max(np.max(train_data[:, 1]), np.max(test_data[:, 1])) + 1

    print('--- Loading data and building ---')
    init_logging_and_result(args)
    graph = tf.Graph()
    with graph.as_default():
        if args.fix_seed:
            tf.set_random_seed(1)
        model = MODEL(args, num_user, num_item)
        model.build_graph()
        training(model, args, train_data, val_data, test_data, test_data_neg, num_user, num_item)


if __name__ == '__main__':
    run()
