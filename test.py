'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import numpy as np
import torch
import argparse
import os
import cv2
from datasets import create_dataset
from models import build_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK')
    parser.add_argument('--serial_batches', action='store_true', help='Disable random shuffling and use sequential batch sampling if enabled')
    parser.add_argument('--num_threads', default=1, type=int, help='Number of subprocesses for data loading')
    parser.add_argument('--phase', type=str, default='test', help='Runtime phase selector')
    parser.add_argument('--load_width', type=int, default=512, help='Input image width for preprocessing (will be resized)')
    parser.add_argument('--load_height', type=int, default=512, help='Input image height for preprocessing (will be resized)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of samples per batch')
    parser.add_argument('--device', default='cuda', help='Computation device [cuda|cpu] for training/inference')
    parser.add_argument('--dataset_mode', type=str, default='crack', help='Dataset mode selector')
    parser.add_argument('--dataset_path', default="../data/TUT", help='Root directory path for dataset')
    parser.add_argument('--model_file_path', default="../data/TUT", help='Root directory path for checkpoint file')
    parser.add_argument('--result_save_path', default="../data/TUT", help='Root directory path for test results')

    args = parser.parse_args()

    t_all = []
    device = torch.device(args.device)
    test_dl = create_dataset(args)
    data_size = len(test_dl)
    model, criterion = build_model(args)
    state_dict = torch.load(args.model_file_path)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("Load Model Successful!")
    save_root = args.result_save_path
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    with torch.no_grad():
        model.eval()
        for batch_idx, (data) in enumerate(test_dl):
            x = data["image"]
            target = data["label"]
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)

            target = target[0, 0, ...].cpu().numpy()
            out = out[0, 0, ...].cpu().numpy()
            root_name = data["A_paths"][0].split("/")[-1][0:-4]
            target = 255 * (target / np.max(target))
            out = 255 * (out / np.max(out))

            # out[out >= 0.5] = 255
            # out[out < 0.5] = 0

            # print('----------------------------------------------------------------------------------------------')
            # print(os.path.join(save_root, "{}_lab.png".format(root_name)))
            # print(os.path.join(save_root, "{}_pre.png".format(root_name)))
            # print('----------------------------------------------------------------------------------------------')
            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)

    print("Finished!")
