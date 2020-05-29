from config import eval_config as config
import tensorflow as tf
import time
import numpy as np
import logging
import argparse
import os
import json
import glob
from utils import _parse_function
from multiprocessing import Pool


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

############################## load lookups ################################
N = config.n_classes

lookup = np.zeros([config.R,config.n_classes]).astype(int)
inv_lookup = np.zeros([config.R,config.n_classes]).astype(int)
counts = np.zeros([config.R,config.B+1]).astype(int)
for r in range(config.R):
    lookup[r] = np.load(config.lookups_loc+'bucket_order_'+str(r)+'.npy')[:N]
    inv_lookup[r] = np.load(config.lookups_loc+'class_order_'+str(r)+'.npy')[:N] 
    counts[r] = np.load(config.lookups_loc+'counts_'+str(r)+'.npy')[:config.B+1] 

query_lookup = np.empty([config.R, config.feat_dim_orig], dtype=int)
for r in range(config.R):
    query_lookup[r] = np.load(config.query_lookups_loc+'bucket_order_'+str(r)+'.npy')

##################################
W1 = [None for r in range(config.R)]
b1 = [None for r in range(config.R)]
hidden_layer = [None for r in range(config.R)]
W2 = [None for r in range(config.R)]
b2 = [None for r in range(config.R)]
logits = [None for r in range(config.R)]
probs = [None for r in range(config.R)]
top_buckets = [None for i in range(config.R)]

##################################
params = [np.load(config.model_loc+'r_'+str(r)+'_epoch_'+str(config.eval_epoch)+'.npz') for r in range(config.R)]
W1_tmp = [params[r]['W1'] for r in range(config.R)]
b1_tmp = [params[r]['b1'] for r in range(config.R)]
W2_tmp = [params[r]['W2'] for r in range(config.R)]
b2_tmp = [params[r]['b2'] for r in range(config.R)]

################# Data Loader ####################
eval_files = glob.glob(config.tfrecord_loc+'*_test.tfrecords')

dataset = tf.data.TFRecordDataset(eval_files)
dataset = dataset.apply(tf.contrib.data.map_and_batch(
    map_func=_parse_function, batch_size=config.batch_size))

iterator = dataset.make_initializable_iterator()
next_y_idxs, next_y_vals, next_x_idxs, next_x_vals = iterator.get_next()
x = [tf.SparseTensor(tf.stack([next_x_idxs.indices[:,0], tf.gather(query_lookup[r], next_x_idxs.values)], axis=-1),
    next_x_vals.values, [config.batch_size, config.feat_hash_dim]) for r in range(config.R)]

############################## Create Graph ################################
for r in range(config.R):
    with tf.device('/gpu:'+str(r//config.R_per_gpu)): 
        ######
        # W1[r] = tf.Variable(tf.truncated_normal([config.feat_hash_dim, config.hidden_dim], stddev=0.05, dtype=tf.float32))
        # b1[r] = tf.Variable(tf.truncated_normal([config.hidden_dim], stddev=0.05, dtype=tf.float32))
        # hidden_layer[r] = tf.nn.relu(tf.sparse_tensor_dense_matmul(x[r],W1[r])+b1[r])
        # #
        # W2[r] = tf.Variable(tf.truncated_normal([config.hidden_dim, config.B], stddev=0.05, dtype=tf.float32))
        # b2[r] = tf.Variable(tf.truncated_normal([config.B], stddev=0.05, dtype=tf.float32))
        ######
        W1[r] = tf.constant(W1_tmp[r])
        b1[r] = tf.constant(b1_tmp[r])
        hidden_layer[r] = tf.nn.relu(tf.sparse_tensor_dense_matmul(x[r],W1[r])+b1[r])
        #
        W2[r] = tf.constant(W2_tmp[r])
        b2[r] = tf.constant(b2_tmp[r])
        ######
        logits[r] = tf.matmul(hidden_layer[r],W2[r])+b2[r]
        # probs[r] = tf.sigmoid(logits[r])
        top_buckets[r] = tf.nn.top_k(logits[r], k=config.topk, sorted=True)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())


################# Load Eval Files #####################
n_check = 1000
count = 0
score_sum = [0.0,0.0,0.0]

##### Run Graph Optimizer on first batch (might take ~50s) ####
sess.run(iterator.initializer)
top_buckets_, y_idxs = sess.run([top_buckets, next_y_idxs])

###### Re-initialize the data loader ####
sess.run(iterator.initializer)


def process_scores(inp):
    R = inp.shape[0]
    topk = inp.shape[2]
    ##
    scores = {}
    freqs = {}
    ##
    for r in range(config.R):
        for k in range(topk):
            val = inp[r,0,k]
            ##
            for key in inv_lookup[r,counts[r,int(inp[r,1,k])]:counts[r,int(inp[r,1,k])+1]]:
                if key in scores:
                    scores[key] += val
                    freqs[key] += 1  
                else:
                    scores[key] = val
                    freqs[key] = 1
    ##
    i = 0
    while True:
        candidates = np.array([key for key in scores if freqs[key]>=config.minfreq-i])
        if len(candidates)>=5:
            break
        i += 1
    scores = np.array([scores[key] for key in candidates])
    # ##
    top_idxs = np.argpartition(scores, -5)[-5:]
    temp = np.argsort(-scores[top_idxs])
    return candidates[top_idxs[temp]]

p = Pool(config.n_cores)

begin_time = time.time()

with open(config.logfile, 'a', encoding='utf-8') as fw:
    while True:
        try:
            top_buckets_, y_idxs = sess.run([top_buckets, next_y_idxs])
            top_buckets_ = np.array(top_buckets_)
            top_buckets_ = np.transpose(top_buckets_, (2,0,1,3))
            preds = p.map(process_scores, top_buckets_)
            ##
            curr_batch_size = y_idxs[2][0]
            labels = [[] for i in range(curr_batch_size)]
            for j in range(len(y_idxs[0])):
                labels[y_idxs[0][j,0]].append(y_idxs[1][j])
            ##
            for i in range(curr_batch_size):
                true_labels = labels[i]
                #### P@1
                if preds[i][0] in true_labels:
                    score_sum[0] += 1
                #### P@3
                score_sum[1] += len(np.intersect1d(preds[i][:3],true_labels))/min(len(true_labels),3)
                #### P@5
                score_sum[2] += len(np.intersect1d(preds[i],true_labels))/min(len(true_labels),5)
                count += 1
            if count%n_check==0:
                print('P@1 for',count,'points:',score_sum[0]/count, file=fw)
                print('P@3 for',count,'points:',score_sum[1]/count, file=fw)
                print('P@5 for',count,'points:',score_sum[2]/count, file=fw)
                print('time_elapsed: ',time.time()-begin_time, file=fw)
        except tf.errors.OutOfRangeError:
            print('overall P@1 for',count,'points:',score_sum[0]/count, file=fw)
            print('overall P@3 for',count,'points:',score_sum[1]/count, file=fw)
            print('overall P@5 for',count,'points:',score_sum[2]/count, file=fw)
            print('time_elapsed: ',time.time()-begin_time, file=fw)
            break

p.close()
