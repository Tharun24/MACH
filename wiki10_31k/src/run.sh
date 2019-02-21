##### Enabling the import of a function used in evaluation #####
cd ../../util/
sudo make
export PYTHONPATH=$(pwd)

##### Move into the respective folder #####
cd ../wiki10_31k/src

#####  Build lookups for classes #####
python3 preproc.py
python3 build_index.py

##### Training multiple repetitions simulataneously #####
tmux new -d -s 0 'python3 train_single.py --repetition=0 --B=2000 --gpu=0 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 1 'python3 train_single.py --repetition=1 --B=2000 --gpu=0 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 2 'python3 train_single.py --repetition=2 --B=2000 --gpu=0 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 3 'python3 train_single.py --repetition=3 --B=2000 --gpu=0 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 4 'python3 train_single.py --repetition=4 --B=2000 --gpu=1 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 5 'python3 train_single.py --repetition=5 --B=2000 --gpu=1 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 6 'python3 train_single.py --repetition=6 --B=2000 --gpu=1 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 7 'python3 train_single.py --repetition=7 --B=2000 --gpu=1 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 8 'python3 train_single.py --repetition=8 --B=2000 --gpu=2 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 9 'python3 train_single.py --repetition=9 --B=2000 --gpu=2 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 10 'python3 train_single.py --repetition=10 --B=2000 --gpu=2 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 11 'python3 train_single.py --repetition=11 --B=2000 --gpu=2 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 12 'python3 train_single.py --repetition=12 --B=2000 --gpu=3 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 13 'python3 train_single.py --repetition=13 --B=2000 --gpu=3 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 14 'python3 train_single.py --repetition=14 --B=2000 --gpu=3 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 15 'python3 train_single.py --repetition=15 --B=2000 --gpu=3 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 16 'python3 train_single.py --repetition=16 --B=2000 --gpu=4 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 17 'python3 train_single.py --repetition=17 --B=2000 --gpu=4 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 18 'python3 train_single.py --repetition=18 --B=2000 --gpu=4 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 19 'python3 train_single.py --repetition=19 --B=2000 --gpu=4 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 20 'python3 train_single.py --repetition=20 --B=2000 --gpu=5 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 21 'python3 train_single.py --repetition=21 --B=2000 --gpu=5 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 22 'python3 train_single.py --repetition=22 --B=2000 --gpu=5 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 23 'python3 train_single.py --repetition=23 --B=2000 --gpu=5 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 24 'python3 train_single.py --repetition=24 --B=2000 --gpu=6 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 25 'python3 train_single.py --repetition=25 --B=2000 --gpu=6 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 26 'python3 train_single.py --repetition=26 --B=2000 --gpu=6 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 27 'python3 train_single.py --repetition=27 --B=2000 --gpu=6 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 28 'python3 train_single.py --repetition=28 --B=2000 --gpu=7 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 29 'python3 train_single.py --repetition=29 --B=2000 --gpu=7 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 30 'python3 train_single.py --repetition=30 --B=2000 --gpu=7 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'
tmux new -d -s 31 'python3 train_single.py --repetition=31 --B=2000 --gpu=7 --gpu_usage=0.2 --batch_size=100 --n_epochs=60 --load_epoch=0'

##### Get precision@1,3,5 #####
python3 eval.py --R=32