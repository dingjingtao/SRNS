import os
import pdb
import tensorflow as tf
import numpy as np
from time import time
import pickle
import sys


class MODEL():
    def __init__(self, args, num_user, num_item):
        self.learning_rate = args.lr
        self.opt = args.optimizer
        self.regs = float(args.regs)
        self.batch_size=args.batch_size
        self.model_file = args.model_file
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_size = args.embedding_size
        self.use_pretrain = args.use_pretrain


    def _create_placeholders(self):
        self.user_input=tf.placeholder(tf.int32,shape=[None,1],name="user_input")
        self.item_input_pos=tf.placeholder(tf.int32,shape=[None,1],name="item_input_pos")
        self.item_input_neg=tf.placeholder(tf.int32,shape=[None,1],name="item_input_neg")
        self.example_weight=tf.placeholder(tf.float32,shape=[None,1],name="example_weight")


    def _create_variables(self, model):
        if not self.use_pretrain:
            self.embeddingmap_user = tf.Variable(
                    tf.truncated_normal(shape=[self.num_user, self.embedding_size], mean=0.0, stddev=0.01),
                                        name='embedding_user', dtype=tf.float32)
            self.embeddingmap_item = tf.Variable(
                    tf.truncated_normal(shape=[self.num_item, self.embedding_size], mean=0.0, stddev=0.01),
                                        name='embedding_item', dtype=tf.float32)
            self.h = tf.Variable(tf.random_uniform([self.embedding_size, 1], minval=-tf.sqrt(6 / (self.embedding_size + 1)),maxval=tf.sqrt(6 / (self.embedding_size + 1))), name='h')
        
        else:
            param=pickle.load(open('./model/model_%s.pkl'%(self.model_file),'rb'))
            self.embeddingmap_user = tf.get_variable(name="embedding_user", initializer=tf.constant(param[0]))
            self.embeddingmap_item = tf.get_variable(name="embedding_item", initializer=tf.constant(param[1]))
            self.h = tf.get_variable(name="h", initializer=tf.constant(param[2]))
              

        
    def _create_loss(self):
        embedding_user = tf.nn.embedding_lookup(self.embeddingmap_user,self.user_input)
        embedding_user = tf.reshape(embedding_user,[-1,self.embedding_size])

        embedding_item_pos = tf.nn.embedding_lookup(self.embeddingmap_item,self.item_input_pos)
        embedding_item_pos = tf.reshape(embedding_item_pos,[-1,self.embedding_size])

        embedding_item_neg = tf.nn.embedding_lookup(self.embeddingmap_item,self.item_input_neg)
        embedding_item_neg = tf.reshape(embedding_item_neg,[-1,self.embedding_size])

        self.score = tf.reshape(tf.matmul(embedding_user * embedding_item_pos, self.h, name='output'),[-1,1])
        self.score_neg = tf.reshape(tf.matmul(embedding_user * embedding_item_neg, self.h, name='output_neg'),[-1,1])

        self.regularizer = tf.contrib.layers.l2_regularizer(self.regs)
        if self.regs!=0.0:
            self.loss_reg = self.regularizer(embedding_user)+self.regularizer(embedding_item_pos)+self.regularizer(embedding_item_neg)
        else:
            self.loss_reg = 0

        self.loss_vanilla = tf.reduce_sum(self.example_weight * tf.log(1 + tf.exp(self.score_neg - self.score - 1e-8)))
        self.loss = self.loss_vanilla + self.loss_reg


    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,name='adam_opt').minimize(self.loss)


    def build_graph(self):
        graph = tf.get_default_graph()
        with graph.as_default() as g:
            with g.name_scope("GMF"):
                self._create_placeholders()
                self._create_variables(self.model_file)
                self._create_loss()
                self._create_optimizer()
       

    def save_model_withpath(self,sess,id):
        param = []
        embedding=sess.run([self.embeddingmap_user, self.embeddingmap_item, self.h])
        for each in embedding:
            param.append(each)
        pickle.dump(param,open(id,'wb'))




