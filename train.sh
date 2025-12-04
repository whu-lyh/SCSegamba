#!/usr/bin/env bash
clear
GPUS="0"

# # Dataset:Crack500
# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --dataset_path /workspace/Data/CrackSeg/Crack500 \
#                                 --output_dir /workspace/WorkSpaceMamba/SCSegamba/experiments/ \
#                                 --epochs 50 \
#                                 --batch_size_train 12 \
#                                 --batch_size_test 12 \
#                                 --num_threads 12

# # Dataset:DeepCrack
# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --dataset_path /workspace/Data/CrackSeg/DeepCrack \
#                                 --output_dir /workspace/WorkSpaceMamba/SCSegamba/experiments/ \
#                                 --epochs 50 \
#                                 --batch_size_train 12 \
#                                 --batch_size_test 12 \
#                                 --num_threads 12 

# # Dataset:CrackMap
# CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
#                                 --dataset_path /workspace/Data/CrackSeg/CrackMap \
#                                 --output_dir /workspace/WorkSpaceMamba/SCSegamba/experiments/ \
#                                 --epochs 50 \
#                                 --batch_size_train 12 \
#                                 --batch_size_test 12 \
#                                 --num_threads 12 

# Dataset:TUT
CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
                                --dataset_path /workspace/Data/CrackSeg/TUT \
                                --output_dir /workspace/WorkSpaceMamba/SCSegamba/experiments/ \
                                --epochs 50 \
                                --batch_size_train 12 \
                                --batch_size_test 12 \
                                --num_threads 12 