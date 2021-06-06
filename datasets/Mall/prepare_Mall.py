import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

dataset = 'Mall'
parts = ['train', 'test']  # train / test
# maxSize = 1024  # (w, h)
# minSize = 512  # (w, h)

workspace_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
data_path = '/root/workspace/datasets/Mall/'
output_path = os.path.join(workspace_path, 'ProcessedData', dataset)

if os.path.isdir(workspace_path):
    sys.path.append(workspace_path)
    from datasets.get_density_map_gaussian import get_density_map_gaussian
if not os.path.exists(output_path):
    os.makedirs(output_path)

for part in parts:
    print('Processing {:s} data of {:s} dataset'.format(part, dataset))

    data_path_img = os.path.join(data_path, '{:s}_data/frames/'.format(part))
    data_path_gt = os.path.join(data_path, '{:s}_label/labels/'.format(part))

    output_path_dir = os.path.join(output_path, '{:s}/'.format(part))
    output_path_img = os.path.join(output_path_dir, 'img/')
    output_path_den = os.path.join(output_path_dir, 'den/')

    if not os.path.exists(output_path_dir):
        os.makedirs(output_path_dir)
    if not os.path.exists(output_path_img):
        os.makedirs(output_path_img)
    if not os.path.exists(output_path_den):
        os.makedirs(output_path_den)

    num_samples = len(os.listdir(data_path_img))

    for idx in range(1, num_samples + 1):
        i = idx if part == 'train' else idx + 800
        # load gt
        gt_fname = os.path.join(data_path_gt, 'seq_{:06d}.txt'.format(i))
        # gt_mat = loadmat(gt_fname)
        # gt = gt_mat['image_info'][0][0][0][0][0]
        gt = pd.read_csv(gt_fname, sep='&', header=None).values
        gt = gt.astype(np.float32, copy=False)

        img_fname = os.path.join(data_path_img, 'seq_{:06d}.jpg'.format(i))
        img = cv2.imread(img_fname)
        [h, w, c] = img.shape

        # resize
        w_ = (int(w / 64)) * 64
        # if w_ > 1024:
        #     w_ = 1024
        # elif w_ < 384:
        #     w_ = 384

        h_ = (int(h / 64)) * 64
        # if h_ > 1024:
        #     h_ = 1024
        # elif h_ < 384:
        #     h_ = 384

        img_ = cv2.resize(img, (w_, h_))
        rate_w, rate_h = float(w_) / w, float(h_) / h
        gt[:, 0] = gt[:, 0] * float(rate_w)
        gt[:, 1] = gt[:, 1] * float(rate_h)

        # generation
        dm = get_density_map_gaussian(img_, gt, 15, 4)

        # save files
        cv2.imwrite(os.path.join(output_path_img, '{:d}.jpg'.format(i)), img_)
        dm = pd.DataFrame(dm)
        dm.to_csv(os.path.join(output_path_den, '{:d}.csv'.format(i)),
                  header=False, index=False)
