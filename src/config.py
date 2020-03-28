class train_config:
    train_data_loc = '/home/trm3/MACH_improved/data/Amazon-3M/'
    tfrecord_loc = '/home/trm3/MACH_improved/data/Amazon-3M/tfrecords/'
    model_save_loc = '/home/trm3/MACH_improved/saved_models/Amazon-3M/b_10000/'
    query_lookups_loc = '/home/trm3/MACH_improved/lookups/Amazon-3M/b_100000/'
    lookups_loc = '/home/trm3/MACH_improved/lookups/Amazon-3M/b_10000/'
    logfile = '/home/trm3/MACH_improved/logs/Amazon-3M/b_10000/'
    ####
    feat_dim_orig = 337067
    n_classes = 2812281
    ####
    n_cores = 4 # core count for parallelizable operations
    B = 10000
    batch_size = 2000
    n_epochs = 10
    load_epoch = 0
    feat_hash_dim = 100000
    hidden_dim = 4096
    # Only used if training multiple repetitions from the same script
    R_per_gpu = 2

class eval_config:
    query_lookups_loc = '/home/trm3/MACH_improved/lookups/Amazon-3M/b_100000/'
    lookups_loc = '/home/trm3/MACH_improved/lookups/Amazon-3M/b_10000/'
    model_loc = '/home/trm3/MACH_improved/saved_models/Amazon-3M/b_10000/'
    eval_data_loc = '/home/trm3/MACH_improved/data/Amazon-3M/'
    tfrecord_loc = '/home/trm3/MACH_improved/data/Amazon-3M/tfrecords/'
    logfile = '/home/trm3/MACH_improved/logs/Amazon-3M/eval_logs_10K_top50_minfreq2.txt'
    ###
    feat_dim_orig = 337067
    n_classes = 2812281
    ###
    B = 10000
    R = 4
    eval_epoch = 10
    R_per_gpu = 4
    num_gpus = 1 # R/R_per_gpu gpus
    n_cores = 48 # core count for parallelizable operations
    batch_size = 2000
    feat_hash_dim = 100000
    hidden_dim = 4096
    ### For approx eval
    topk = 50 # how many top buckets to take
    minfreq = 2 # min number of times a class  
