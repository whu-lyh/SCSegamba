#!/usr/bin/env bash
clear
GPUS="0"

CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
                                --dataset_path /workspace/Data/CrackSeg/TUT \
                                --model_file_path /workspace/WorkSpaceMamba/SCSegamba/checkpoints/weights/2025_10_19_10:45:13_Dataset->TUT/checkpoint_best.pth \
                                --result_save_path /workspace/WorkSpaceMamba/SCSegamba/results/2025_10_19_10:45:13_Dataset->TUT 
