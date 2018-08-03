#!/bin/bash
python main.py --batch_size 256 --momentum 0.9 --weight_decay 1e-4 --data_path /data/l-dataset/ --dataset l-bird \
               --pretrain --print_freq 1 --epochs 32 --schedule 15 23 --newfc --numclass_old 1000 --numclass_new 10320  --workers 64 --resume checkpoints/checkpoint.pth.tar


