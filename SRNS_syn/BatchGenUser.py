import numpy as np
import random
import pdb
import pickle
def sampling(args,train_data):    
    index = list(range(train_data.shape[0]))
    np.random.shuffle(index)
    num_batch = train_data.shape[0] // args.batch_size
    user_batch_all = []
    item_batch_all = []
    for batch_idx in range(num_batch):
        begin = batch_idx * args.batch_size
        item_batch = []
        user_batch = []
        for idx in range(begin,begin+args.batch_size):
            user_batch.append(train_data[index[idx],0])
            item_batch.append(train_data[index[idx],1])
        user_batch_all.append(user_batch)
        item_batch_all.append(item_batch)
    return user_batch_all, item_batch_all




