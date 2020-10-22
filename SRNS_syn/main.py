import os
import tensorflow as tf
import numpy as np
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
    parser.add_argument('--process_name', nargs='?', default='Ecom-toy_exp',
                        help='Input process name.')
    parser.add_argument('--model', nargs='?', default='uniform',
                        help='uniform/SRNS')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1000,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--regs', nargs='?', default='0.0',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--embedding_size', type=int, default=32,
                        help='embedding_size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--gpu', type=str, default='4',
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
    parser.add_argument('--S1', type=int, default=20,
                        help='')
    parser.add_argument('--S2', type=int, default=20,
                        help='')
    parser.add_argument('--fix_seed', action='store_true', default=False,
                        help='fix random seed')
    parser.add_argument('--model_file', type=str, default="./model/model.pkl",
                        help='model file path')
    parser.add_argument('--save_model_filename', type=str, default="./model/model.pkl",
                        help='model file path')
    parser.add_argument('--fn_num', type=int, default=1,
                        help='1')
    parser.add_argument('--sigma', type=int, default=2,
                        help='2/4/6/8/10')
    parser.add_argument('--dynamic_alpha', type=str, default='no',
                        help='no/increase/decrease')
    parser.add_argument('--T0', type=int, default=100,
                        help='50/100')
    parser.add_argument('--dataset', type=int, default=2,
                        help='1/2')
    return parser.parse_args()



def training(model, args, train_data, test_data, test_data_merge, fn_data, num_user, num_item, num_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        logging.info("--- Start training ---%s"%filename)
        print("--- Start training ---%s"%filename)
        sess.run(tf.global_variables_initializer())

        metric_best = 0
        stop_counter = 0

        train_set = dict()
        for i in range(train_data.shape[0]):
            if train_data[i,0] not in train_set:
                train_set[train_data[i,0]] = set()
            train_set[train_data[i,0]].add(train_data[i,1])

        test_set = dict()
        for i in range(test_data.shape[0]):
            if test_data[i,0] not in test_set:
                test_set[test_data[i,0]] = set()
            test_set[test_data[i,0]].add(test_data[i,1])

        for i in range(num_user):
            if i not in train_set:
                train_set[i]=set()
            if i not in test_set:
                test_set[i]=set()
        
        fn_num_list=[]
        Mu = []
        for i in range(num_user):
            if args.model=="uniform":
                continue
            Mu.append([])
            if args.fn_num!=0:
                fn_num_list.append(min(args.fn_num,len(fn_data[i])))
                if len(fn_data[i])<=args.fn_num:
                    Mu[i]+=fn_data[i]
                else:
                    Mu[i]+=random.sample(fn_data[i],args.fn_num)
            while len(Mu[i])!=args.S1:
                random_item=random.randint(0,num_item-1)
                while random_item in train_set[i] or random_item in test_set[i]:
                    random_item=random.randint(0,num_item-1)
                Mu[i].append(random_item)


        RECALL, NDCG, score_one_epoch = EvalUser.eval(model, sess, train_set, test_data_merge, num_user, num_item)
        print("Before trianing, RECALL = %.4f/%.4f/%.4f, NDCG = %.4f/%.4f/%.4f" % (RECALL[0], RECALL[1], RECALL[2], NDCG[0], NDCG[1], NDCG[2]))
        logging.info("Before trianing, RECALL = %.4f/%.4f/%.4f, NDCG = %.4f/%.4f/%.4f" % (RECALL[0], RECALL[1], RECALL[2], NDCG[0], NDCG[1], NDCG[2]))
        score_all = np.array([score_one_epoch])

        for epoch_count in range(args.epochs):
            if epoch_count == 0:
                variance_all = np.zeros((num_user,num_item))
            else:
                if args.model=="uniform" or epoch_count<5:
                    variance_all = np.zeros((num_user,num_item))
                else:
                    mean_all = np.mean(score_all[-5:,:,:],axis=0)
                    variance_all = np.zeros(mean_all.shape)
                    for i in range(5):
                        variance_all += (score_all[i-5,:,:]-mean_all)**2
                    variance_all = np.sqrt(variance_all/5)

            train_begin = time()
            batches = BatchUser.sampling(args,train_data)
            loss, Mu = training_batch(model, sess, batches, args, epoch_count, train_data, fn_data, train_set, test_set, fn_num_list, num_item, score_all[epoch_count,:,:], variance_all, Mu,epoch_count)
            train_time = time() - train_begin

            RECALL, NDCG, score_one_epoch = EvalUser.eval(model, sess, train_set, test_data_merge, num_user, num_item)
            print ("Epoch %d [%.1fs]: loss=%.4f, RECALL = %.4f/%.4f/%.4f, NDCG = %.4f/%.4f/%.4f" % (epoch_count+1, train_time, loss, RECALL[0], RECALL[1], RECALL[2], NDCG[0], NDCG[1], NDCG[2]))
            logging.info("Epoch %d [%.1fs]: loss=%.4f, RECALL = %.4f/%.4f/%.4f, NDCG = %.4f/%.4f/%.4f" % (epoch_count+1, train_time, loss, RECALL[0], RECALL[1], RECALL[2], NDCG[0], NDCG[1], NDCG[2]))
            score_all = np.concatenate([score_all,np.array([score_one_epoch])], axis=0)
            
            if RECALL[2] > metric_best:
                stop_counter = 0
                metric_best = RECALL[2]
                if args.save_model:
                    model.save_model(sess, args.save_model_filename)
                    print("Model Saved")
            else:
                stop_counter += 1
                if stop_counter > args.early_stop:
                    print ("early stopped")
                    logging.info("early stopped")
                    break



def training_batch(model,sess, batches, args, epoch_cur, train_data, fn_data, train_set, test_set, fn_num_list, num_item, score_one_epoch, variance_all, Mu, epoch_count):
    user_batch, item_batch = batches
    num_batch = len(user_batch)
    loss_average = 0.0

    for batch_idx in range(num_batch):
        if args.model == "uniform":
            negitems_candidates = np.random.choice(num_item, [len(user_batch[batch_idx]),1])
            for i in range(len(user_batch[batch_idx])):
                while negitems_candidates[i,0] in train_set[user_batch[batch_idx][i]] : 
                    negitems_candidates[i,0] = random.randint(0,num_item-1)
            negitems = np.reshape(negitems_candidates,[-1,1])
            users = np.reshape(np.array(user_batch[batch_idx]),[-1,1])
            positems = np.reshape(np.array(item_batch[batch_idx]),[-1,1])
            feed_dict = {
                model.user_input:users,
                model.item_input_pos:positems,
                model.item_input_neg:negitems}
            _, loss = sess.run([model.optimizer, model.loss],feed_dict)
            loss_average += loss        
        else:
            negitems = []
            for i in range(len(user_batch[batch_idx])):
                ratings_candidates=[]
                ratings_candidates2=[]
               
                negitems_candidates2=Mu[user_batch[batch_idx][i]]
                for j in range(len(negitems_candidates2)):
                    if args.dynamic_alpha=='increase':
                        ratings_candidates2.append(score_one_epoch[user_batch[batch_idx][i],negitems_candidates2[j]]
                                                    +min(1,epoch_count/args.T0)*args.alpha*variance_all[user_batch[batch_idx][i],negitems_candidates2[j]])
                    elif args.dynamic_alpha=='decrease':
                        ratings_candidates2.append(score_one_epoch[user_batch[batch_idx][i],negitems_candidates2[j]]
                                                    +max(0,1-epoch_count/args.T0)*args.alpha*variance_all[user_batch[batch_idx][i],negitems_candidates2[j]])
                    else:
                        ratings_candidates2.append(score_one_epoch[user_batch[batch_idx][i],negitems_candidates2[j]]
                                                    +args.alpha*variance_all[user_batch[batch_idx][i],negitems_candidates2[j]])
                ratings_candidates2 = np.array(ratings_candidates2)
                negitems.append(negitems_candidates2[np.argmax(ratings_candidates2)])   
            
                #update Mu
                if len(fn_data[user_batch[batch_idx][i]])>args.fn_num: 
                    Mu[user_batch[batch_idx][i]][0:args.fn_num]=random.sample(list(fn_data[user_batch[batch_idx][i]]),args.fn_num)
                while len(Mu[user_batch[batch_idx][i]]) < args.S2+args.S1:
                    random_item = random.randint(0,num_item-1)
                    while random_item in train_set[user_batch[batch_idx][i]] or random_item in test_set[user_batch[batch_idx][i]]:
                        random_item = random.randint(0,num_item-1)
                    Mu[user_batch[batch_idx][i]].append(random_item)
                if args.fn_num!=0:
                    start_pos=min(fn_num_list[user_batch[batch_idx][i]],args.fn_num)
                else:
                    start_pos=0
                for j in range(start_pos,args.S2+args.S1):
                    ratings_candidates.append(score_one_epoch[user_batch[batch_idx][i],Mu[user_batch[batch_idx][i]][j]])
                ratings_candidates = np.array(ratings_candidates)/args.temperature
                ratings_candidates = np.exp(ratings_candidates)/np.sum(np.exp(ratings_candidates))
                Mu_items_arg = np.random.choice(args.S2+args.S1-start_pos, args.S1-start_pos, p=ratings_candidates, replace=False)+start_pos
                Mu[user_batch[batch_idx][i]][start_pos:]=np.array(Mu[user_batch[batch_idx][i]])[Mu_items_arg].tolist()
                    
            users = np.reshape(user_batch[batch_idx],[-1,1])
            positems = np.reshape(item_batch[batch_idx],[-1,1])
            negitems = np.reshape(np.array(negitems),[-1,1])
            feed_dict = {model.user_input:users,
                        model.item_input_pos:positems,
                        model.item_input_neg:negitems}
            _, loss = sess.run([model.optimizer, model.loss],feed_dict)
            loss_average += loss
    loss_average /= train_data.shape[0]
    return loss_average, Mu     
    
            

def init_logging_and_result(args):
    Log_dir_name = 'Log'
    if not os.path.exists(Log_dir_name):
        os.makedirs(Log_dir_name)
    F_model=args.model.lower()
    F_trail_id = args.trial_id
    F_optimizer = args.optimizer
    F_lr = str(args.lr)
    F_reg=args.regs
    F_alpha=str(args.alpha)
    F_temp = str(args.temperature)
    global filename
    filename = "log-%s-%s-lr%s-reg%s-alpha%s-temp%s-S1_%s-S2_%s-num_%d-sigma_%d-%s" % (
    F_model, F_optimizer, F_lr, F_reg, F_alpha, F_temp,str(args.S1),str(args.S2), args.fn_num, args.sigma, F_trail_id)

    if args.dynamic_alpha!='no':
        filename+="dynamic_alpha-"+args.dynamic_alpha+'-'+str(args.T0)
    if not os.path.exists(Log_dir_name + '/' + filename):
        logging.basicConfig(filename=Log_dir_name + '/' + filename, level=logging.INFO)
    else:
        print(Log_dir_name + '/' + filename, 'already exists, skipping ...')


def run():
    args = parse_args()
    if args.fix_seed:
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)
        print("Fix Random Seed!!!")
   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    setproctitle.setproctitle(args.process_name)

    if args.dataset==1:
        train_data = pickle.load(open('./ml-100k_tuning/train.pkl','rb'))
        test_data = pickle.load(open('./ml-100k_tuning/test.pkl','rb'))
        test_data_merge = pickle.load(open('./ml-100k_tuning/test_merge.pkl','rb'))
        fn_data = pickle.load(open('./ml-100k_tuning/fn_sigma_%d.pkl'%(args.sigma),'rb'))
    elif args.dataset==2:
        train_data = pickle.load(open('./ml-100k/train.pkl','rb'))
        test_data = pickle.load(open('./ml-100k/test.pkl','rb'))
        test_data_merge = pickle.load(open('./ml-100k/test_merge.pkl','rb'))
        fn_data = pickle.load(open('./ml-100k/fn_sigma_%d.pkl'%(args.sigma),'rb'))
    else:
        print("Wrong dataset setting!")
        exit()
   
    num_train = train_data.shape[0]
    num_user = max(np.max(train_data[:,0]),np.max(test_data[:,0]))+1
    num_item = max(np.max(train_data[:,1]),np.max(test_data[:,1]))+1

    init_logging_and_result(args)
    model = MODEL(args, num_user, num_item)
    model.build_graph()
    training(model, args, train_data, test_data, test_data_merge, fn_data, num_user, num_item, num_train)



if __name__ == '__main__':
    run()




