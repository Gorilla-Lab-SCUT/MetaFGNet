#!/bin/bash

python main.py --epochs 100540 --schedule 50270 75405 1801300 1801300 1801300 --data-path-source /data/l-dataset/ --data-path /data/CUB_200_2011/ \
              --auxiliary-dataset l_bird --dataset cub200 --num-classes-s 10320 --num-classes-t 200 --arch resnet34 --num-updates-for-gradient 1 \
              --meta-train-lr 0.01 --batch-size 256 --batch-size-source 256 --test-freq 4968 --record-freq 500 --print-freq 1 --lr 0.1 --log From_noselection_secondOrd_nofirst_update \
              --pretrained --pretrained-checkpoint ../MetaFGNet_without_Sample_Selection/From_24epoch_secondOrd_noFirstUpdate/model_best.pth.tar  --workers 16  --second-order-grad \
                ## len(selected auxiliary) = 2543856, the number of training iterations is depending on the number of training images.
               #--meta-sgd --first-meta-update
