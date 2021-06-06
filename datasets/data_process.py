import cv2
import numpy as np
import torch
import pandas as pd
from torchvision.transforms.functional import resize, crop
from torchvision import transforms


def random_crop(img, den, crop_size, label_factor=1):
    _, src_h, src_w = img.shape
    crop_h, crop_w = crop_size

    dst_h, dst_w = src_h, src_w

    img = transforms.ToPILImage()(img)
    den = transforms.ToPILImage()(den)

    if (src_h, src_w) < (crop_h, crop_w):
        resize_rate_h = 1.0 * crop_h / src_h if src_h < crop_h else 1.0
        resize_rate_w = 1.0 * crop_w / src_w if src_w < crop_w else 1.0
        resize_rate = max(resize_rate_h, resize_rate_w)
        dst_h, dst_w = round(src_h * resize_rate), round(src_w * resize_rate)

        img = resize(img, size=(dst_h, dst_w))
        den = resize(den, size=(dst_h, dst_w))

    i = np.random.randint(0, dst_h - crop_h) // label_factor * label_factor
    j = np.random.randint(0, dst_w - crop_w) // label_factor * label_factor

    label_i = i // label_factor
    label_j = j // label_factor

    img_crop = transforms.ToTensor()(crop(img, i, j, crop_h, crop_w))
    den_crop = transforms.ToTensor()(crop(den, label_i, label_j, crop_h, crop_w))

    return img_crop, den_crop


def share_memory(batch, is_share=False):
    out = None
    if is_share:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out


def get_min_size(batch, min_size=None):
    min_ht = min_size[0]
    min_wd = min_size[1]

    for i_sample in batch:
        _, ht, wd = i_sample.shape
        # resize the image to fit the crop size
        if [wd, ht] < [min_wd, min_ht]:
            # resize_rate =
            pass
        if ht < min_ht:
            min_ht = ht
        if wd < min_wd:
            min_wd = wd
    return min_ht, min_wd
