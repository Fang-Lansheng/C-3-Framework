import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps

import pandas as pd

from config import cfg


class UCSD(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        # ~
        # self.img_path = data_path + '/img'
        # self.gt_path = data_path + '/den'
        # ~
        self.img_path = data_path + '_data'
        self.gt_path = data_path + '_label'
        # ~
        data_files = []
        for folder in os.listdir(self.img_path):
            data_files += [filename for filename in os.listdir(os.path.join(self.img_path, folder))
                           if os.path.isfile(os.path.join(self.img_path, folder, filename))]
        self.data_files = data_files
        # self.data_files = [filename for filename in os.listdir(self.img_path)
        #                    if os.path.isfile(os.path.join(self.img_path, filename))]
        # ~
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        self.mode = mode

        if self.mode == 'train':
            print('[UCSD DATASET]: %d training images.' % (self.num_samples))
        if self.mode == 'test':
            print('[UCSD DATASET]: %d testing images.' % (self.num_samples))

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        folder = fname.split('_f')[0] + '.y'
        img = Image.open(os.path.join(self.img_path, folder, fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        den_path = os.path.join(self.gt_path, folder, os.path.splitext(fname)[0] + '.txt')
        den = pd.read_csv(den_path, sep='&', header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples
