## Introduction
This repository has the official code for the algorithm MACH discussed in the NeurIPS 2019 paper [Extreme Classification in Log-Memory using Count-Min Sketch](https://papers.nips.cc/paper/9482-extreme-classification-in-log-memory-using-count-min-sketch-a-case-study-of-amazon-search-with-50m-products.pdf). 
MACH proposes a zero-communication distributed training method for Extreme Classification (classification with millions of classes). 

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
or 
```
@article{medini2019extreme,
  title={Extreme Classification in Log Memory using Count-Min Sketch: A Case Study of Amazon Search with 50M Products},
  author={Medini, Tharun and Huang, Qixuan and Wang, Yiqiu and Mohan, Vijai and Shrivastava, Anshumali},
  journal={arXiv preprint arXiv:1910.13830},
  year={2019}
}
```

## Download links for datasets
Most of the public datasets that we use are available on [Extreme Classification Repository (XML Repo)](http://manikvarma.org/downloads/XC/XMLRepository.html). Specific links are as follows:

1. [Amazon-670K] (https://drive.google.com/file/d/1TLaXCNB_IDtLhk4ycOnyud0PswWAW6hR/view) - [Kaggle Link](https://www.kaggle.com/c/extreme-classification-amazon) 
2. [Delicious-200K] (https://drive.google.com/file/d/0B3lPMIHmG6vGR3lBWWYyVlhDLWM/view)
3. [Wiki10-31K] (http://manikvarma.org/downloads/XC/XMLRepository.html) 

After downloading any of the XML repo datasets, please unzip them and move the train and test files to any folder(s) of your choice. Update *train_data_loc* and *eval_data_loc* in *config.py*.

4. ODP dataset: [Train] (http://hunch.net/~vw/odp_train.vw.gz) / [Test] (http://hunch.net/~vw/odp_test.vw.gz) . 
The data format must be changed to match the datasets on Extreme Classification repo.

5. Fine-grained ImageNet-22K dataset: [Train] (http://hunch.net/~jl/datasets/imagenet/training.txt.gz) / [Test](http://hunch.net/~jl/datasets/imagenet/testing.txt.gz) .
Yet again, the data format must be changed to match the datasets on Extreme Classification repo.

## Running MACH

# requirements
You are expected to have TensorFlow 1.x installed (1.8 - 1.14 should work) and have atleast 2 GPUs with 32GB memory (or 4 GPUs with 16 GB memory). We will add support for TensorFlow 2.x in subsequent versions. 
*Cython* is also required for importing a C++ function *gather_batch* during evaluation (if you cannot use C++ for any reason, please refer to the **C++ vs Python for evaluation** section below).
*sklearn* is required for importing *murmurhash3_32* (from sklearn.utils). Although the version requirements for *cython* and *sklearn* are non that stringent as Tensorflow, 
use Cython-0.29.14 and sklearn-0.22.2 in case you run into any issues.

# configuration
After cloning the repo, move in to *src* folder and change the config.py file. Most of the configurations are self explanatory. Some non-trivial ones are:
1. *feat_dim_orig* corresponds to the original input dimension of the dataset. Since this might be huge for some datasets, we need to feature hash it to smaller dimension (set by *feat_hash_dim*)
2. *feat_hash_dim* is the smaller dimension that the input is hashed into. To avoid loss of information, we use different random seeds for hash functions in each independent model. 
3. *lookups_loc* is the location to save murmurhash lookups for each model (each lookup is an *n_classes* dimensional integer array with each value ranging from [0,B-1]).
4. *B* stands for number of buckets (*B*<<*n_classes*)
5. *R* in eval_config is the number of models that we want to use for evaluation.
6. *R_per_gpu* stands for how many models can simultaneously be run on a single GPU. We can generally run upto 8 models (each with around 400M parameters) at once on a single V-100 GPU with 32 GB memory.   

# preprocessing 
We are going to use TFRecords format for the input data (we will soon add support for loading from txt files with tf.data). TFRecords allows the data to be streamed with provision for prefetching and pseudo-shuffling. 
Compared to writing a data loader to load from a plain .txt file, TFRecords reduces GPU idle time and speeds up the trainign by 2-3x. 

The code *preprocess.py* has 3 sub-parts. First one transforms the .txt data to TFRecords format. Second one creates lookups for classes. Third one creates lookups for input feature hashing.

# training
The script *train.sh* has all the commands for running 16 models in parallel via tmux sessions. To run just one model, please run the following command.

```
python3 train.py --repetition=0 --gpu=0 --gpu_usage=0.45
```

*gpu_usage* limits the proportion of GPU memory that this model can use. By limiting it to 45%, we can train 2 models at once on each GPU.

# evaluation

## Some finer details

# C++ vs Python for evaluation (batch size effect)
The code evaluate.py imports a cython function *gather_batch* to get reconstruct the scores  

# tf.constant vs tf.Variable

# logits vs probabilities

# TF Record support

# 