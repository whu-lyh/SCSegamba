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
# CUDA_VISIBLE_DEVICES=${GPUS} python eval_compute.py --model_mode SAVSS

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
# CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
#                                 --dataset_path /workspace/Data/CrackSeg/CrackMap \
#                                 --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_10_20_05_32_14_Dataset_CrackMap/weights/checkpoint_best.pth" \
#                                 --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_10_20_05_32_14_Dataset_CrackMap/weights/epoch36"
# cd eval
# CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_10_20_05_32_14_Dataset_CrackMap/weights/epoch36"
# cd ..

# without_DAU test

# # Dataset:Crack500
# CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
#                                 --dataset_path /workspace/Data/CrackSeg/Crack500 \
#                                 --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_15_43_18_Crack500_wo_direction_aware_updating/weights/checkpoint_best.pth" \
#                                 --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_15_43_18_Crack500_wo_direction_aware_updating/weights/best_test_results"
# cd eval
# CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_15_43_18_Crack500_wo_direction_aware_updating/weights/best_test_results"
# cd ..

# # Dataset:DeepCrack
# CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
#                                 --dataset_path /workspace/Data/CrackSeg/DeepCrack \
#                                 --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_17_07_18_DeepCrack_wo_direction_aware_updating/weights/checkpoint_best.pth" \
#                                 --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_17_07_18_DeepCrack_wo_direction_aware_updating/weights/best_test_results"
# cd eval
# CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_17_07_18_DeepCrack_wo_direction_aware_updating/weights/best_test_results"
# cd ..

# # Dataset:CrackMap
# CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
#                                 --dataset_path /workspace/Data/CrackSeg/CrackMap \
#                                 --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_08_48_54_CrackMap_wo_direction_aware_updating/weights/checkpoint_best.pth" \
#                                 --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_08_48_54_CrackMap_wo_direction_aware_updating/weights/best_test_results"
# cd eval
# CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_08_48_54_CrackMap_wo_direction_aware_updating/weights/best_test_results"
# cd ..

# # Dataset:TUT
# CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
#                                 --dataset_path /workspace/Data/CrackSeg/TUT \
#                                 --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_08_13_14_TUT_wo_direction_aware_updating/weights/checkpoint_best.pth" \
#                                 --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_08_13_14_TUT_wo_direction_aware_updating/weights/best_test_results"
# cd eval
# CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_05_08_13_14_TUT_wo_direction_aware_updating/weights/best_test_results"
# cd ..



# Dataset:Crack500
CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
                                --model_mode HSMM \
                                --use_noisy_gate \
                                --use_residual_connection \
                                --dataset_path /workspace/Data/CrackSeg/Crack500 \
                                --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_16_34_41_Crack500_HSMMv1/weights/checkpoint_best.pth" \
                                --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_16_34_41_Crack500_HSMMv1/weights/best_test_results"
cd eval
CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_16_34_41_Crack500_HSMMv1/weights/best_test_results"
cd ..

# Dataset:DeepCrack
CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
                                --model_mode HSMM \
                                --use_noisy_gate \
                                --use_residual_connection \
                                --dataset_path /workspace/Data/CrackSeg/DeepCrack \
                                --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_18_27_39_DeepCrack_HSMMv1/weights/checkpoint_best.pth" \
                                --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_18_27_39_DeepCrack_HSMMv1/weights/best_test_results"
cd eval
CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_18_27_39_DeepCrack_HSMMv1/weights/best_test_results"
cd ..

# Dataset:CrackMap
CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
                                --model_mode HSMM \
                                --use_noisy_gate \
                                --use_residual_connection \
                                --dataset_path /workspace/Data/CrackSeg/CrackMap \
                                --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_18_47_01_CrackMap_HSMMv1/weights/checkpoint_best.pth" \
                                --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_18_47_01_CrackMap_HSMMv1/weights/best_test_results"
cd eval
CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_18_47_01_CrackMap_HSMMv1/weights/best_test_results"
cd ..

# Dataset:TUT
CUDA_VISIBLE_DEVICES=${GPUS} python test.py \
                                --model_mode HSMM \
                                --use_noisy_gate \
                                --use_residual_connection \
                                --dataset_path /workspace/Data/CrackSeg/TUT \
                                --model_file_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_18_52_53_TUT_HSMMv1/weights/checkpoint_best.pth" \
                                --result_save_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_18_52_53_TUT_HSMMv1/weights/best_test_results"
cd eval
CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py --result_path "/workspace/WorkSpaceMamba/SCSegamba/experiments/2025_12_22_18_52_53_TUT_HSMMv1/weights/best_test_results"
cd ..
