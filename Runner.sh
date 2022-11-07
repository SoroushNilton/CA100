#!/bin/bash


for i in 1
do
        python3 main.py --dataset imagenet --data_dir '/media/myHardDrive/Research/my_mini_imagenet' --epochs 15 --train_batch_size 120 --eval_batch_size 90 --job_dir './result/resnet_50/hrank' --resume '/home/soroush/Desktop/Maintainable/Pre_Trained_Dir/resnet50-19c8e357.pth' --arch resnet_50 --gpu '0'
        python Hrank_ResNet50.py
done