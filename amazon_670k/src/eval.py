import os
import sys
import numpy as np
import tensorflow as tf
import time
import glob
import json
import argparse
import math
from multiprocessing import Pool

try:
    from util import gather_batch
    from util import gather_K
except:
    print('**********************CANNOT IMPORT GATHER***************************')
    exit()

parser = argparse.ArgumentParser()
parser.add_argument("--R", help="how many repetitions?", default=32, type=int)
parser.add_argument("--R_per_gpu", help="how many repetitions per GPU", default=8, type=int)
parser.add_argument("--B", help="how many buckets?", default=10000, type=int)
parser.add_argument("--gpu", help="which GPU?", default='0,1,2,3', type=str)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--epoch", default='40', type=str)
parser.add_argument("--parallel_across", default='batch', choices=['batch','classes'], type=str)
parser.add_argument("--n_threads", default=16, type=int)
args = parser.parse_args()

if not args.gpu=='all':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

## Training Params
feature_dim = 135909
hidden_dim_1 = 500
hidden_dim_2 = 500
B = args.B
batch_size = args.batch_size
epoch = args.epoch

R = args.R

R_per_gpu = args.R_per_gpu
if R%R_per_gpu==0:
    num_gpus = R//R_per_gpu
else:
    num_gpus = R//R_per_gpu + 1

num_classes = 670091

lookup = np.empty([R,num_classes], dtype=int)
for r in range(R):
    lookup[r] = np.load('../data/b_'+str(B)+'/lookups/bucket_order_'+str(r)+'.npy')

params = [None for r in range(R)]
for r in range(R):
    params[r] = np.load('../saved_models/b_'+str(B)+'/r_'+str(r)+'_epoch_'+epoch+'.npz')

x_idxs = tf.placeholder(tf.int64, shape=[None,2])
x_vals = tf.placeholder(tf.float32, shape=[None])
x = tf.SparseTensor(x_idxs, x_vals, [batch_size,feature_dim])

W1_tmp=np.array([params[r]['weights_1'] for r in range(R)])
b1_tmp=np.array([params[r]['bias_1'] for r in range(R)])
W2_tmp=np.array([params[r]['weights_2'] for r in range(R)])
b2_tmp=np.array([params[r]['bias_2'] for r in range(R)])
W3_tmp=np.array([params[r]['weights_3'] for r in range(R)])
b3_tmp=np.array([params[r]['bias_3'] for r in range(R)])

W1 = [None for i in range(R)]
b1 = [None for i in range(R)]
layer_1 = [None for i in range(R)]
W2 = [None for i in range(R)]
b2 = [None for i in range(R)]
layer_2 = [None for i in range(R)]
W3 = [None for i in range(R)]
b3 = [None for i in range(R)]
logits = [None for i in range(R)]
probs = [None for i in range(R)]

for i in range(num_gpus):
    with tf.device('/gpu:'+str(i)):
        for r in range(R_per_gpu*i,min(R_per_gpu*(i+1),R)):
            W1[r] = tf.Variable(W1_tmp[r])
            b1[r] = tf.Variable(b1_tmp[r])
            layer_1[r] = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1[r])+b1[r])
            #
            W2[r] = tf.Variable(W2_tmp[r])
            b2[r] = tf.Variable(b2_tmp[r])
            layer_2[r] = tf.nn.relu(tf.matmul(layer_1[r],W2[r])+b2[r])
            #
            W3[r] = tf.Variable(W3_tmp[r])
            b3[r] = tf.Variable(b3_tmp[r])
            logits[r] = tf.matmul(layer_2[r],W3[r])+b3[r]

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

###########################################################
N = num_classes
candidates = np.array(range(num_classes))
candidate_indices = np.ascontiguousarray(lookup[:,candidates])

begin_time = time.time()
with open('../data/test.txt','r',encoding='utf-8') as f:
    idxs = []
    vals = []
    labels = []
    count = 0
    offset = 0
    score_sum = [0.0, 0.0, 0.0]
    for line in f:
        try:
            itms = line.strip().split()
            labels.append([int(lbl) for lbl in itms[0].split(',')]) 
            idxs += [(count-offset,int(itm.split(':')[0])) for itm in itms[1:]]
            vals += [float(itm.split(':')[1]) for itm in itms[1:]]    
            count += 1
            if count%batch_size==0:
                output = sess.run(logits,feed_dict={x_idxs:idxs, x_vals:vals})
                #
                preds = np.reshape(output,[R,batch_size,B])
                preds = np.transpose(preds, (1,0,2))
                preds = np.ascontiguousarray(preds)
                #
                scores = np.zeros((batch_size,N), dtype=np.float32)
                if args.parallel_across=='batch':
                    gather_batch(preds, candidate_indices, scores, R, B, N, batch_size, args.n_threads)
                else:
                    gather_K(preds, candidate_indices, scores, R, B, N, batch_size, args.n_threads)
                ########
                top_lbls_1 = np.argmax(scores, axis=-1)
                top_lbls_3 = np.argpartition(scores, -3, axis=-1)[:, -3:]
                top_lbls_5 = np.argpartition(scores, -5, axis=-1)[:, -5:]
                for i in range(batch_size):
                    #### P@1
                    if top_lbls_1[i] in labels[i]:
                        score_sum[0] += 1
                    #### P@3
                    score_sum[1] += len(np.intersect1d(top_lbls_3[i],labels[i]))/min(len(labels[i]),3)
                    #### P@5
                    score_sum[2] += len(np.intersect1d(top_lbls_5[i],labels[i]))/min(len(labels[i]),5)
                #
                idxs = []
                vals = []
                labels = []
                offset = count
            #
            if count%1000==0:
                print('precision@1 for',count,'points:',score_sum[0]/count)
                print('precision@3 for',count,'points:',score_sum[1]/count)
                print('precision@5 for',count,'points:',score_sum[2]/count)
                print('time_elapsed: ',time.time()-begin_time)
        except KeyboardInterrupt:
            print('precision@1 for',count,'points:',score_sum[0]/count)
            print('precision@3 for',count,'points:',score_sum[1]/count)
            print('precision@5 for',count,'points:',score_sum[2]/count)
            print('time_elapsed: ',time.time()-begin_time)
            break

print('precision@1 for all',count,'points with '+str(R)+' repetitions:',score_sum[0]/count)
print('precision@3 for all',count,'points with '+str(R)+' repetitions:',score_sum[1]/count)
print('precision@5 for all',count,'points with '+str(R)+' repetitions:',score_sum[2]/count)
print('overall time_elapsed: ',time.time()-begin_time)

