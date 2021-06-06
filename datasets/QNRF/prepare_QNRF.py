import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

dataset = 'QNRF'
parts = ['train', 'test']  # train / test
maxSize = 1024  # (w, h)
minSize = 512  # (w, h)

workspace_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
data_path = '/root/workspace/datasets/UCF-QNRF/'
output_path = os.path.join(workspace_path, 'ProcessedData', dataset)

if os.path.isdir(workspace_path):
    sys.path.append(workspace_path)
    from datasets.get_density_map_gaussian import get_density_map_gaussian
if not os.path.exists(output_path):
    os.makedirs(output_path)

for part in parts:
    print('Processing {:s} data of {:s} dataset'.format(part, dataset))

    data_path_img = os.path.join(data_path, '{:s}'.format(part))
    data_path_gt = os.path.join(data_path, '{:s}'.format(part))

    output_path_dir = os.path.join(output_path, '{:s}/'.format(part))
    output_path_img = os.path.join(output_path_dir, 'img/')
    output_path_den = os.path.join(output_path_dir, 'den/')

    if not os.path.exists(output_path_dir):
        os.makedirs(output_path_dir)
    if not os.path.exists(output_path_img):
        os.makedirs(output_path_img)
    if not os.path.exists(output_path_den):
        os.makedirs(output_path_den)

    for d in os.listdir(data_path_img):
        for f in os.listdir(os.path.join(data_path_img, d)):
            img_path = os.path.join(data_path_img, d, f)
            gt_path = os.path.join(data_path_gt, d, f.split('.')[0] + '.txt')

            gt = pd.read_csv(gt_path, sep='&', header=None).values
            gt = gt.astype(np.float32, copy=False)

            img = cv2.imread(img_path)
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

            # generation
            img_ = cv2.resize(img, (w_, h_))
            rate_w, rate_h = float(w_) / w, float(h_) / h
            gt[:, 0] = gt[:, 0] * float(rate_w)
            gt[:, 1] = gt[:, 1] * float(rate_h)

            dm = get_density_map_gaussian(img_, gt, 15, 4)

            # save files
            fname = '{:03d}_{:03d}'.format(int(d), int(f.split('.')[0]))
            cv2.imwrite(os.path.join(output_path_img, '{:s}.jpg'.format(fname)), img_)
            dm = pd.DataFrame(dm)
            dm.to_csv(os.path.join(output_path_den, '{:s}.csv'.format(fname)),
                      header=False, index=False)
