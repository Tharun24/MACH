# Introduction
This repository has the official code for the algorithm MACH discussed in the NeurIPS 2019 paper [Extreme Classification in Log-Memory using Count-Min Sketch]
(https://papers.nips.cc/paper/9482-extreme-classification-in-log-memory-using-count-min-sketch-a-case-study-of-amazon-search-with-50m-products.pdf). 
MACH proposes a novel zero-communication distributed training method for Extreme Classification (classification with millions of classes). We project the huge output 
vector with millions of dimensions to a small dimensional count-min sketch (CMS) matrix. We then train indpenedent networks to predict each column of this CMS matrix 
instead of the hige label vector.

If you find our approach interesting, please cite our paper with the following bibtex
```
@inproceedings{medini2019extreme,
  title={Extreme Classification in Log Memory using Count-Min Sketch: A Case Study of Amazon Search with 50M Products},
  author={Medini, Tharun Kumar Reddy and Huang, Qixuan and Wang, Yiqiu and Mohan, Vijai and Shrivastava, Anshumali},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13244--13254},
  year={2019}
}
```

# Download links for datasets
Most of the public datasets that we use are available on [Extreme Classification Repository (XML Repo)](http://manikvarma.org/downloads/XC/XMLRepository.html). Specific links are as follows:

1. [Amazon-670K] (https://drive.google.com/file/d/1TLaXCNB_IDtLhk4ycOnyud0PswWAW6hR/view) / [Kaggle Link](https://www.kaggle.com/c/extreme-classification-amazon) 
2. [Delicious-200K] (https://drive.google.com/file/d/0B3lPMIHmG6vGR3lBWWYyVlhDLWM/view)
3. [Wiki10-31K] (http://manikvarma.org/downloads/XC/XMLRepository.html) 

After downloading any of the XML repo datasets, please unzip them and move the train and test files to any folder(s) of your choice. Update *train_data_loc* and *eval_data_loc* in *config.py*.

4. ODP dataset: [Train] (http://hunch.net/~vw/odp_train.vw.gz) / [Test] (http://hunch.net/~vw/odp_test.vw.gz) . 
The data format must be changed to match the datasets on Extreme Classification repo.

5. Fine-grained ImageNet-22K dataset: [Train] (http://hunch.net/~jl/datasets/imagenet/training.txt.gz) / [Test](http://hunch.net/~jl/datasets/imagenet/testing.txt.gz) .
Yet again, the data format must be changed to match the datasets on Extreme Classification repo.

# Running MACH

## Requirements
You are expected to have TensorFlow 1.x installed (1.8 - 1.14 should work) and have atleast 2 GPUs with 32GB memory (or 4 GPUs with 16 GB memory). We will add support for TensorFlow 2.x in subsequent versions. 
*Cython* is also required for importing a C++ function *gather_batch* during evaluation (if you cannot use C++ for any reason, please refer to the **Cython vs Python for evaluation** section below).
*sklearn* is required for importing *murmurhash3_32* (from sklearn.utils). Although the version requirements for *cython* and *sklearn* are non that stringent as Tensorflow, 
use Cython-0.29.14 and sklearn-0.22.2 in case you run into any issues.

## Configuration
After cloning the repo, move in to *src* folder and change the config.py file. Most of the configurations are self explanatory. Some non-trivial ones are:
1. *feat_dim_orig* corresponds to the original input dimension of the dataset. Since this might be huge for some datasets, we need to feature hash it to smaller dimension (set by *feat_hash_dim*)
2. *feat_hash_dim* is the smaller dimension that the input is hashed into. To avoid loss of information, we use different random seeds for hash functions in each independent model. 
3. *lookups_loc* is the location to save murmurhash lookups for each model (each lookup is an *n_classes* dimensional integer array with each value ranging from [0,B-1]).
4. *B* stands for number of buckets (*B*<<*n_classes*)
5. *R* in eval_config is the number of models that we want to use for evaluation.
6. *R_per_gpu* stands for how many models can simultaneously be run on a single GPU. We can generally run upto 8 models (each with around 400M parameters) at once on a single V-100 GPU with 32 GB memory.   

## Pre-processing 
We are going to use TFRecords format for the input data (we will soon add support for loading from txt files with tf.data). TFRecords allows the data to be streamed with provision for prefetching and pseudo-shuffling. 
Compared to writing a data loader to load from a plain .txt file, TFRecords reduces GPU idle time and speeds up the trainign by 2-3x. 

The code *preprocess.py* has 3 sub-parts. First one transforms the .txt data to TFRecords format. Second one creates lookups for classes. Third one creates lookups for input feature hashing.
The line of data is assumed to be in the format *2980,3177,9026,9053,12256,12258 63:5.906890 87:3.700440 242:6.339850 499:2.584960 611:4.321930 672:2.807350*  where 2980,...,12258 are true labels while subsequent 
ones are input feature indexes and their respective values. If you want to parse any other data format, please modify the function *create_tfrecords* in *utils.py*. 

Once you're clear with the data format, please run
```
python3 preprocess.py
```

## Training
The script *train.sh* has all the commands for running 16 models in parallel via tmux sessions. To run just one model, please run the following command.

```
python3 train.py --repetition=0 --gpu=0 --gpu_usage=0.45
```

*gpu_usage* limits the proportion of GPU memory that this model can use. By limiting it to 45%, we can train 2 models at once on each GPU.

## Evaluation
Please run the follwoing commands to compile a Cython function. You may require *sudo* access to your machine for this step. In case you don't have it, read the section *Cython vs Python for evaluation* below.
```
cd src/util/
make clean
make
export PYTHONPATH=$(pwd)
```

After an error-free compilation, please change the config.py file to specify the number of models/repetitions *R*, which epoch's models to evaulate with, data paths, logfile paths etc. 
Adjust the *R_per_gpu* in case you run out of GPU memory. Then run

```
python3 evaluate.py
```

The time taken to load the lookups, weights and initialize the network is quite substantial. It takes >30 mins for 16 models each with ~400M parameters. The time printed in the log files doesn not account for 
the initialization. It only measures the proper evaluation time.

# Some finer details

## Cython vs Python for evaluation
The code evaluate.py imports a cython function *gather_batch* to reconstruct the scores for all classes from the predicted count-min sketch logits. It also gets the top-5 predictions based on these gathered scores 
using a priority queue implementation. We use simple *#pragma omp* parallelization across a batch of inputs.

Alternatively, we can also do the same aggregation and partitioning of scores in python and parallelize it using *multiprocessing* *Pool* object. However, using exclusively python for this step cause the evaluation 
to be more than **5x slower**. Nevertheless, if you cannot use Cython, please comment the *import gather_batch* and running *gather_batch(...)* lines from *evaluate.py* and uncomment the following two snippets:
```
def process_logits(inp):
    R = inp.shape[0]
    B = inp.shape[1]
    ##
    scores = np.zeros(config.n_classes, dtype=float)
    ##
    for r in range(R):
        for b in range(B):
            val = inp[r,b]
            scores[inv_lookup[r, counts[r,b]:counts[r,b+1]]] += val
    ##
    top_idxs = np.argpartition(scores, -5)[-5:]
    temp = np.argsort(-scores[top_idxs])
    return top_idxs[temp]

p = Pool(config.n_cores)
```

and 

```
top_preds = p.map(process_logits, logits_)
```

You should then be able to run
```
python3 evaluate.py
```

**Note:** pragma omp parallel doesn't allow huge matrices to be passed on. Hence adjust the *batch_size* in *eval_config* so that *batch_size*n_classes < 3 billion*.

## tf.constant vs tf.Variable
In *evaluate.py*, the network initialization uses *tf.constant()* when loading saved weights instead of *tf.Variable()* as we do not have to train the weights again (it might be slightly faster too). 
However, some older versions of TensorFlow might not allow initializing a Tensor with a >2GB matrix. In that case, we need to define a *tf.placeholder()* first, then define a *tf.Variable(tf.placeholder())*
and feed the matrix later during variable initialization.

## logits vs probabilities
In the paper, the proposed unbiased estimator for recovering all probabilities is proportional to the sum of predicted bucket probabilities. However, for multilabel classification, it turns out thet summing up 
predicted bucket logits instead of probabilities gives better precision because logits have wide range of values compared to probabilities. Nevertheless, if you want to experiment with probabilities, you can 
simply changes

```
logits_, y_idxs = sess.run([logits, next_y_idxs])
```

to 

```
probs_, y_idxs = sess.run([probs, next_y_idxs])
```

## TF Record support
##