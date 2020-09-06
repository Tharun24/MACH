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

try:
    from util import gather_batch
    from util import gather_K
except:
    print('**********************CANNOT IMPORT GATHER***************************')
    exit()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

############################## load lookups ################################
## inv_lookup and counts are NOT needed if you are using C++ gather function (see further for details)

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

##################### create empty lists for future tensors  ################
W1 = [None for r in range(config.R)]
b1 = [None for r in range(config.R)]
hidden_layer = [None for r in range(config.R)]
W2 = [None for r in range(config.R)]
b2 = [None for r in range(config.R)]
logits = [None for r in range(config.R)]
probs = [None for r in range(config.R)]
top_buckets = [None for i in range(config.R)]

####################### load saved weights ############################
#### If you just want to test the code for bugs, don't load these.
#### Just use random weight when creating a TF graph (shown later)

params = [np.load(config.model_loc+'r_'+str(r)+'_epoch_'+str(config.eval_epoch)+'.npz') for r in range(config.R)]
W1_tmp = [params[r]['W1'] for r in range(config.R)]
b1_tmp = [params[r]['b1'] for r in range(config.R)]
W2_tmp = [params[r]['W2'] for r in range(config.R)]
b2_tmp = [params[r]['b2'] for r in range(config.R)]

################# Create TF Data Loader ####################
eval_files = glob.glob(config.tfrecord_loc+'*_test.tfrecords')

dataset = tf.data.TFRecordDataset(eval_files)
dataset = dataset.apply(tf.contrib.data.map_and_batch(
    map_func=_parse_function, batch_size=config.batch_size, num_parallel_calls=4))
# dataset = dataset.prefetch(buffer_size=10)
iterator = dataset.make_initializable_iterator()
next_y_idxs, next_y_vals, next_x_idxs, next_x_vals = iterator.get_next()
x = [tf.SparseTensor(tf.stack([next_x_idxs.indices[:,0], tf.gather(query_lookup[r], next_x_idxs.values)], axis=-1),
    next_x_vals.values, [config.batch_size, config.feat_hash_dim]) for r in range(config.R)]

############################## Create Graph ################################

#### Uncomment these if you are using placeholders and writing your own data loader
# x_idxs = tf.placeholder(tf.int64, [None, 2])
# # x_vals = tf.ones_like(x_idxs[:,0], dtype=tf.float32)
# x_vals = tf.placeholder(tf.float32, [None,])
# x = tf.SparseTensor(x_idxs, x_vals, [config.batch_size, config.feat_hash_dim])
####


for r in range(config.R):
    with tf.device('/gpu:'+str(r//config.R_per_gpu)): 
        ###### Random weight initialization to test for bugs
        # W1[r] = tf.Variable(tf.truncated_normal([config.feat_hash_dim, config.hidden_dim], stddev=0.05, dtype=tf.float32))
        # b1[r] = tf.Variable(tf.truncated_normal([config.hidden_dim], stddev=0.05, dtype=tf.float32))
        # hidden_layer[r] = tf.nn.relu(tf.sparse_tensor_dense_matmul(x[r],W1[r])+b1[r])
        # #
        # W2[r] = tf.Variable(tf.truncated_normal([config.hidden_dim, config.B], stddev=0.05, dtype=tf.float32))
        # b2[r] = tf.Variable(tf.truncated_normal([config.B], stddev=0.05, dtype=tf.float32))
        ###### Load weights into tensors (tf.constant takes less memory than tf.Variable)
        W1[r] = tf.constant(W1_tmp[r])
        b1[r] = tf.constant(b1_tmp[r])
        hidden_layer[r] = tf.nn.relu(tf.sparse_tensor_dense_matmul(x[r],W1[r])+b1[r])
        #
        W2[r] = tf.constant(W2_tmp[r])
        b2[r] = tf.constant(b2_tmp[r])
        ######
        logits[r] = tf.matmul(hidden_layer[r],W2[r])+b2[r]
        probs[r] = tf.sigmoid(logits[r])


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())

################ (Uncomment this snippet if you want to use python multiprocessing)
# def process_logits(inp):
#     R = inp.shape[0]
#     B = inp.shape[1]
#     ##
#     scores = np.zeros(config.n_classes, dtype=float)
#     ##
#     for r in range(R):
#         for b in range(B):
#             val = inp[r,b]
#             scores[inv_lookup[r, counts[r,b]:counts[r,b+1]]] += val
#     ##
#     top_idxs = np.argpartition(scores, -5)[-5:]
#     temp = np.argsort(-scores[top_idxs])
#     return top_idxs[temp]

# p = Pool(config.n_cores)
#################

################# Evaluation begins #####################
n_check = 100
count = 0
overall_count = 0
score_sum = [0.0,0.0,0.0]

begin_time = time.time()

sess.run(iterator.initializer) # initialize TF data loader

with open(config.logfile, 'a', encoding='utf-8') as fw:
    while True:
        try:
            logits_, y_idxs = sess.run([logits, next_y_idxs])
            logits_ = np.array(logits_)
            logits_ = np.transpose(logits_, (1,0,2))
            logits_ = np.ascontiguousarray(logits_)
            curr_batch_size = y_idxs[2][0]
            ## C++ gather function (faster)
            scores = np.zeros([curr_batch_size, N], dtype=np.float32)
            top_preds = np.zeros([curr_batch_size, 5], dtype=np.int64)
            gather_batch(logits_, lookup, scores, top_preds, config.R, config.B, N, curr_batch_size, config.n_cores)
            ## python multiprocessing (~5x slower than C++ gather)
            # top_preds = p.map(process_logits, logits_)
            ## get true labels
            labels = [[] for i in range(curr_batch_size)]
            for j in range(len(y_idxs[0])):
                labels[y_idxs[0][j,0]].append(y_idxs[1][j])
            ##
            for i in range(curr_batch_size):
                true_labels = labels[i]
                sorted_preds = top_preds[i]
                #### P@1
                if sorted_preds[0] in true_labels:
                    score_sum[0] += 1
                #### P@3
                score_sum[1] += len(np.intersect1d(sorted_preds[:3],true_labels))/min(len(true_labels),3)
                #### P@5
                score_sum[2] += len(np.intersect1d(sorted_preds,true_labels))/min(len(true_labels),5)
                count += 1
            if count%n_check==0:
                print('P@1 for',count,'points:',score_sum[0]/count, file=fw)
                print('P@3 for',count,'points:',score_sum[1]/count, file=fw)
                print('P@5 for',count,'points:',score_sum[2]/count, file=fw)
                print('time_elapsed: ',time.time()-begin_time, file=fw)
        except tf.errors.OutOfRangeError: # this happens after the last batch of data is loaded
            print('overall P@1 for',count,'points:',score_sum[0]/count, file=fw)
            print('overall P@3 for',count,'points:',score_sum[1]/count, file=fw)
            print('overall P@5 for',count,'points:',score_sum[2]/count, file=fw)
            print('time_elapsed: ',time.time()-begin_time, file=fw)
            break

