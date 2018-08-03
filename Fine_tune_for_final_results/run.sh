#!/bin/bash
## The result of MetaFGNet without sample selection
#python main.py --epochs 160 --schedule 79 119 --lr 0.1 --print-freq 10 --resume ../MetaFGNet_without_Sample_Selection/From_24epoch_secondOrd_nofirst_update_resnet34_cub200_256Timg_l_bird_256Simg_Meta_train_Lr0.01_1/model_best.pth.tar  --data-path /data/CUB_200_2011/ --dataset cub200 --num-classes-t 200 --num-classes-s 10320 --log checkpoints_metafgnet_lBird_firstIter --new-fc --batch_size 128 ## prec1=87.17

## The result of MetaFGNet with sample selection
# python main.py --epochs 160 --schedule 79 119 --lr 0.1 --print-freq 10 --resume ../MetaFGNet_with_Sample_Selection/From_noselection_secondOrd_nofirst_update_resnet34_cub200_256Timg_l_bird_256Simg_Meta_train_Lr0.01_1/model_best.pth.tar  --data-path /data/CUB_200_2011/ --dataset cub200 --num-classes-t 200 --num-classes-s 10320 --log checkpoints_metafgnet_lBird_secondIter --new-fc --batch_size 128

