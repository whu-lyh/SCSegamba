#!/usr/bin/env bash
clear
GPUS="0"

# Official pretrained models test

# Dataset:TUT
# CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
#                                 --dataset_path /workspace/Data/CrackSeg/TUT \
#                                 --model_file_path /workspace/WorkSpaceMamba/SCSegamba/pretrained_models/checkpoint_TUT.pth \
#                                 --result_save_path /workspace/WorkSpaceMamba/SCSegamba/pretrained_models/results_TUT

# FLOPS AND PARAMS
# CUDA_VISIBLE_DEVICES=${GPUS} python eval_compute.py 

# the metric calculation seems odd cause the middle results are saved in local disks, which may occupy large storage
# cd eval
# CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path /workspace/WorkSpaceMamba/SCSegamba/pretrained_models/results_TUT
# cd ..


# # Dataset:TUT
# CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
#                                 --dataset_path /workspace/Data/CrackSeg/TUT \
#                                 --model_file_path "./experiments/2025_10_20_05_35_44_Dataset_TUT/weights/checkpoint_best.pth" \
#                                 --result_save_path "./experiments/2025_10_20_05_35_44_Dataset_TUT/weights/epoch49"
# cd eval
# CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_10_20_05_35_44_Dataset_TUT/weights/epoch49"
# cd ..


# # Dataset:CrackMap
CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
                                --dataset_path /workspace/Data/CrackSeg/CrackMap \
                                --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_10_20_05_32_14_Dataset_CrackMap/weights/checkpoint_best.pth" \
                                --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_10_20_05_32_14_Dataset_CrackMap/weights/epoch36"
cd eval
CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_10_20_05_32_14_Dataset_CrackMap/weights/epoch36"
cd ..
