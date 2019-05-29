##### Enabling the import of a function used in evaluation #####
cd ../../util/
sudo make
export PYTHONPATH=$(pwd)

##### Move into the respective folder #####
cd ../delicious_200k/src

#####  Build lookups for classes #####
python3 build_index.py

##### Training multiple repetitions simulataneously #####
tmux new -d -s 0 'python3 train_single.py --repetition=0 --B=5000 --gpu=0 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=1 --B=5000 --gpu=0 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=2 --B=5000 --gpu=0 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=3 --B=5000 --gpu=0 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0'

tmux new -d -s 1 'python3 train_single.py --repetition=4 --B=5000 --gpu=1 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=5 --B=5000 --gpu=1 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=6 --B=5000 --gpu=1 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=7 --B=5000 --gpu=1 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0'

tmux new -d -s 2 'python3 train_single.py --repetition=8 --B=5000 --gpu=2 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=9 --B=5000 --gpu=2 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=10 --B=5000 --gpu=2 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=11 --B=5000 --gpu=2 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0'

tmux new -d -s 3 'python3 train_single.py --repetition=12 --B=5000 --gpu=3 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=13 --B=5000 --gpu=3 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=14 --B=5000 --gpu=3 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=15 --B=5000 --gpu=3 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0'

tmux new -d -s 4 'python3 train_single.py --repetition=16 --B=5000 --gpu=4 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=17 --B=5000 --gpu=4 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=18 --B=5000 --gpu=4 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=19 --B=5000 --gpu=4 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0'

tmux new -d -s 5 'python3 train_single.py --repetition=20 --B=5000 --gpu=5 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=21 --B=5000 --gpu=5 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=22 --B=5000 --gpu=5 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=23 --B=5000 --gpu=5 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0'

tmux new -d -s 6 'python3 train_single.py --repetition=24 --B=5000 --gpu=6 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=25 --B=5000 --gpu=6 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=26 --B=5000 --gpu=6 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=27 --B=5000 --gpu=6 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0'

tmux new -d -s 7 'python3 train_single.py --repetition=28 --B=5000 --gpu=7 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=29 --B=5000 --gpu=7 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=30 --B=5000 --gpu=7 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0;
python3 train_single.py --repetition=31 --B=5000 --gpu=7 --gpu_usage=1.0 --batch_size=1000 --n_epochs=20 --load_epoch=0'

##### Get precision@1,3,5 #####
python3 eval.py --R=32