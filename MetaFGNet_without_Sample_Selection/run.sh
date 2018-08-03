#!/bin/bash

python main.py --epochs 37263 --schedule 12421 24842 1801300 1801300 1801300 --data-path-source /data/l-dataset/ --data-path /data/CUB_200_2011/ \
              --auxiliary-dataset l_bird --dataset cub200 --num-classes-s 10320 --num-classes-t 200 --arch resnet34 --num-updates-for-gradient 1 \
              --meta-train-lr 0.01 --batch-size 256 --batch-size-source 256 --test-freq 6210 --record-freq 500 --print-freq 1 --second-order-grad --lr 0.1 --log From_24epoch_secondOrd_nofirst_update \
              --pretrained --pretrained-checkpoint ../L_Bird_pretrain/checkpoints/24_checkpoint.pth.tar --workers 16 \
              #--resume From_24epoch_secondOrd_nofirst_update_resnet34_cub200_256Timg_l_bird_256Simg_Meta_train_Lr0.01_1/37260checkpoint.pth.tar
                ## len(auxiliary) = ???
               #--meta-sgd --first-meta-update
#bestacc85.795

#### used for test only on the target dataset
#python main.py --epochs 99368 --schedule 49684 74526 1801300 1801300 1801300 --data-path-source /data/l-dataset/ --data-path /data/CUB_200_2011/ \
#              --auxiliary-dataset l_bird --dataset cub200 --num-classes-s 10320 --num-classes-t 200 --arch resnet34 --num-updates-for-gradient 1 \
#              --meta-train-lr 0.01 --batch-size 256 --batch-size-source 256 --test-freq 6210 --record-freq 500 --print-freq 1 --second-order-grad --lr 0.1 --log From_24epoch_secondOrd_nofirst_update \
#              --pretrained --pretrained-checkpoint ../L_Bird_pretrain/checkpoints/24_checkpoint.pth.tar --workers 16 --test-only  \
#              --resume ./60epoch_SecondOrd_nofirst_update_resnet34_cub200_256Timg_l_bird_256Simg_Meta_train_Lr0.01_1/model_best.pth.tar
