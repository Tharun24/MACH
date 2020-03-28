from config import train_config as config
from multiprocessing import Pool
from sklearn.utils import murmurhash3_32 as mmh3
import tensorflow as tf
import glob
import time
import numpy as np
import tensorflow as tf

def create_universal_lookups(r):
    counts = np.zeros(config.B+1, dtype=int)
    bucket_order = np.zeros(config.n_classes, dtype=int)
    #
    for i in range(config.n_classes):
        bucket = mmh3(i,seed=r)%config.B
        bucket_order[i] = bucket
        counts[bucket+1] += 1
    #
    counts = np.cumsum(counts)
    rolling_counts = np.zeros(config.B, dtype=int)
    class_order = np.zeros(config.n_classes,dtype=int)
    for i in range(config.n_classes):
        temp = bucket_order[i]
        class_order[counts[temp]+rolling_counts[temp]] = i
        rolling_counts[temp] += 1
    np.save(config.lookups_loc+'class_order_'+str(r)+'.npy', class_order)
    np.save(config.lookups_loc+'counts_'+str(r)+'.npy',counts)
    np.save(config.lookups_loc+'bucket_order_'+str(r)+'.npy', bucket_order)

def create_query_lookups(r):
    bucket_order = np.zeros(config.feat_dim_orig, dtype=int)
    #
    for i in range(config.feat_dim_orig):
        bucket = mmh3(i,seed=r)%config.feat_hash_dim
        bucket_order[i] = bucket
    np.save(config.query_lookups_loc+'bucket_order_'+str(r)+'.npy', bucket_order)

def input_example(labels, label_vals, inp_idxs, inp_vals): # for writing TFRecords
    labels_list = tf.train.Int64List(value = labels)
    label_vals_list = tf.train.FloatList(value = label_vals)
    inp_idxs_list = tf.train.Int64List(value = inp_idxs)
    inp_vals_list = tf.train.FloatList(value = inp_vals)
    # Create a dictionary with above lists individually wrapped in Feature
    feature = {
        'labels': tf.train.Feature(int64_list = labels_list),
        'label_vals': tf.train.Feature(float_list = label_vals_list),
        'input_idxs': tf.train.Feature(int64_list = inp_idxs_list),
        'input_vals': tf.train.Feature(float_list = inp_vals_list)
    }
    # Create Example object with features
    example = tf.train.Example(features = tf.train.Features(feature=feature))
    return example

def create_tfrecords(file):
    f = open(file, 'r', encoding = 'utf-8')
    header = f.readline()
    write_loc = config.tfrecord_loc+file.split('/')[-1].split('.')[0]
    with tf.python_io.TFRecordWriter(write_loc+'.tfrecords') as writer:
        for line in f:
            itms = line.strip().split()
            y_idxs = [int(itm) for itm in itms[0].split(',')]
            y_vals = [1.0 for itm in range(len(y_idxs))]
            x_idxs = [int(itm.split(':')[0]) for itm in itms[1:]]
            x_vals = [float(itm.split(':')[1]) for itm in itms[1:]]    
            ############################
            tf_example = input_example(y_idxs, y_vals, x_idxs, x_vals)
            writer.write(tf_example.SerializeToString())

def _parse_function(example_proto): # for reading TFRecords
    features = {"labels": tf.VarLenFeature(tf.int64),
              "label_vals": tf.VarLenFeature(tf.float32),
              "input_idxs": tf.VarLenFeature(tf.int64),
              "input_vals": tf.VarLenFeature(tf.float32)
              }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["labels"], parsed_features["label_vals"], parsed_features["input_idxs"], parsed_features["input_vals"]

def _parse_function_eval(example_proto): # for reading TFRecords
    features = {"labels": tf.VarLenFeature(tf.int64),
              "label_vals": tf.VarLenFeature(tf.float32),
              "input_idxs": tf.VarLenFeature(tf.int64),
              "input_vals": tf.VarLenFeature(tf.float32)
              }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["labels"], parsed_features["label_vals"], parsed_features["input_idxs"], parsed_features["input_vals"]
