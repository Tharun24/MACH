from config import train_config as config
import tensorflow as tf
import glob
import argparse
import time
import numpy as np
import logging
from utils import _parse_function

parser = argparse.ArgumentParser()
parser.add_argument("--repetition", help="which repetition?", default=0)
parser.add_argument("--gpu", default=0)
parser.add_argument("--gpu_usage", default=0.45)
args = parser.parse_args()

if not args.gpu=='all':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

r = int(args.repetition) # which repetition

############################## Test code from here ################################
lookup = tf.constant(np.load(config.lookups_loc+'bucket_order_'+str(r)+'.npy'))
query_lookup = tf.constant(np.load(config.query_lookups_loc+'bucket_order_'+str(r)+'.npy'))

train_files = glob.glob(config.tfrecord_loc+'*_train.tfrecords')

dataset = tf.data.TFRecordDataset(train_files)
# dataset = dataset.map(_parse_function, num_parallel_calls=4)
# dataset = dataset.batch(config.batch_size)
dataset = dataset.apply(tf.contrib.data.map_and_batch(
    map_func=_parse_function, batch_size=config.batch_size))
dataset = dataset.prefetch(buffer_size=1000)
dataset = dataset.shuffle(buffer_size=1000)
# dataset = dataset.repeat(config.n_epochs)
iterator = dataset.make_initializable_iterator()
next_y_idxs, next_y_vals, next_x_idxs, next_x_vals = iterator.get_next()
###############
x_idxs = tf.stack([next_x_idxs.indices[:,0], tf.gather(query_lookup, next_x_idxs.values)], axis=-1)
x_vals = next_x_vals.values
x = tf.SparseTensor(x_idxs, x_vals, [config.batch_size, config.feat_hash_dim])
####
y_idxs = tf.stack([next_y_idxs.indices[:,0], tf.gather(lookup, next_y_idxs.values)], axis=-1)
y_vals = next_y_vals.values
y = tf.SparseTensor(y_idxs, y_vals, [config.batch_size, config.B])
y_ = tf.sparse_tensor_to_dense(y, validate_indices=False)
###############
if config.load_epoch>0:
    params=np.load(config.model_save_loc+'r_'+str(r)+'_epoch_'+str(config.load_epoch)+'.npz')
    #
    W1_tmp = tf.placeholder(tf.float32, shape=[config.feat_hash_dim, config.hidden_dim])
    b1_tmp = tf.placeholder(tf.float32, shape=[config.hidden_dim])
    W1 = tf.Variable(W1_tmp)
    b1 = tf.Variable(b1_tmp)
    hidden_layer = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
    #
    W2_tmp = tf.placeholder(tf.float32, shape=[config.hidden_dim, config.B])
    b2_tmp = tf.placeholder(tf.float32, shape=[config.B])
    W2 = tf.Variable(W2_tmp)
    b2 = tf.Variable(b2_tmp)
    logits = tf.matmul(hidden_layer,W2)+b2
else:
    W1 = tf.Variable(tf.truncated_normal([config.feat_hash_dim, config.hidden_dim], stddev=0.05, dtype=tf.float32))
    b1 = tf.Variable(tf.truncated_normal([config.hidden_dim], stddev=0.05, dtype=tf.float32))
    hidden_layer = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
    #
    W2 = tf.Variable(tf.truncated_normal([config.hidden_dim, config.B], stddev=0.05, dtype=tf.float32))
    b2 = tf.Variable(tf.truncated_normal([config.B], stddev=0.05, dtype=tf.float32))
    logits = tf.matmul(hidden_layer,W2)+b2

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_))
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session(config = tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=float(args.gpu_usage))))

if config.load_epoch==0:
    sess.run(tf.global_variables_initializer())
else:
    sess.run(tf.global_variables_initializer(),
        feed_dict = {
            W1_tmp:params['W1'],
            b1_tmp:params['b1'],
            W2_tmp:params['W2'],
            b2_tmp:params['b2']})
    del params

begin_time = time.time()
logging.basicConfig(filename = config.logfile+'logs_'+str(r), level=logging.INFO)
n_check=100

for curr_epoch in range(config.load_epoch+1,config.load_epoch+config.n_epochs+1):
    sess.run(iterator.initializer)
    count = 0
    while True:
        try:
            sess.run(train_op)
            count += 1
            if count%n_check==0:
                _, train_loss = sess.run([train_op, loss])
                logging.info('finished '+str(count)+' steps. Time elapsed for last '+str(n_check)+' steps: '+str(time.time()-begin_time)+' s')
                begin_time = time.time()
                logging.info('train_loss: '+str(train_loss))
                count+=1
        except tf.errors.OutOfRangeError:
            break
    logging.info('###################################')
    logging.info('finished epoch '+str(curr_epoch))
    logging.info('###################################')
    if curr_epoch%5==0:
        params = sess.run([W1,b1,W2,b2])
        np.savez_compressed(config.model_save_loc+'r_'+str(r)+'_epoch_'+str(curr_epoch)+'.npz',
            W1=params[0], 
            b1=params[1], 
            W2=params[2], 
            b2=params[3])
        del params

