import torch
import random
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from datasets.SHHB.SHHB import SHHB
from datasets.SHHB.setting import cfg_data


def get_min_size(batch):
    min_ht = cfg_data.TRAIN_SIZE[0]
    min_wd = cfg_data.TRAIN_SIZE[1]

    for i_sample in batch:
        _, ht, wd = i_sample.shape
        if ht < min_ht:
            min_ht = ht
        if wd < min_wd:
            min_wd = wd
    return min_ht, min_wd


def random_crop(img, den, dst_size):
    # dst_size: ht, wd
    _, ts_hd, ts_wd = img.shape

    x1 = random.randint(0, ts_wd - dst_size[1]) // cfg_data.LABEL_FACTOR * cfg_data.LABEL_FACTOR
    y1 = random.randint(0, ts_hd - dst_size[0]) // cfg_data.LABEL_FACTOR * cfg_data.LABEL_FACTOR
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]

    label_x1 = x1 // cfg_data.LABEL_FACTOR
    label_y1 = y1 // cfg_data.LABEL_FACTOR
    label_x2 = x2 // cfg_data.LABEL_FACTOR
    label_y2 = y2 // cfg_data.LABEL_FACTOR

    return img[:, y1:y2, x1:x2], den[label_y1:label_y2, label_x1:label_x2]


def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out


def SHHB_collate_multibatch(batch):

    transposed = list(zip(*batch))  # imgs and dens
    imgs, dens = [transposed[0], transposed[1]]

    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):
        min_ht, min_wd = get_min_size(imgs)

        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = random_crop(imgs[i_sample], dens[i_sample], [min_ht, min_wd])

            # print('den sum: ', torch.sum(_den, dim=(0, 1)))
            _den = _den.unsqueeze(0)
            _den = _den.unsqueeze(0)
            filter = torch.ones(1, 1, 8, 8, requires_grad=False)
            _den = torch.nn.functional.conv2d(_den, filter, stride=8)
            _den = _den.squeeze()
            # print('den sum: ', torch.sum(_den, dim=(0, 1)))

            cropped_imgs.append(_img)
            cropped_dens.append(_den)
            # print(_den.shape)
            # print(' ')
        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))

        return [cropped_imgs, cropped_dens]

    raise TypeError((error_msg.format(type(batch[0]))))


def SHHB_collate_onebatch(batch):
    transposed = list(zip(*batch))  # imgs and dens
    imgs, dens = [transposed[0], transposed[1]]

    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):

        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = imgs[i_sample], dens[i_sample]

            _den = _den.unsqueeze(0)
            _den = _den.unsqueeze(0)
            filter = torch.ones(1, 1, 8, 8, requires_grad=False)
            _den = torch.nn.functional.conv2d(_den, filter, stride=8)
            _den = _den.squeeze()

            cropped_imgs.append(_img)
            cropped_dens.append(_den)

        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))

        return [cropped_imgs, cropped_dens]

    raise TypeError((error_msg.format(type(batch[0]))))


def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA

    train_main_transform = own_transforms.Compose([
    	#own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
    	own_transforms.RandomHorizontallyFlip()
    ])

    # val_main_transform = own_transforms.Compose([
    #     own_transforms.RandomCrop(cfg_data.TRAIN_SIZE)
    # ])
    val_main_transform = None

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])

    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    # Train loader
    train_set = SHHB(cfg_data.DATA_PATH + '/train', 'train',
                     main_transform=train_main_transform,
                     img_transform=img_transform,
                     gt_transform=gt_transform)

    train_loader = DataLoader(train_set,
                              batch_size=cfg_data.TRAIN_BATCH_SIZE,
                              num_workers=8,
                              collate_fn=SHHB_collate_multibatch,
                              shuffle=True,
                              drop_last=True)


    # Val loader
    val_set = SHHB(cfg_data.DATA_PATH + '/test', 'test',
                   main_transform=val_main_transform,
                   img_transform=img_transform,
                   gt_transform=gt_transform)

    val_loader = DataLoader(val_set,
                            batch_size=cfg_data.VAL_BATCH_SIZE,
                            num_workers=8,
                            collate_fn=SHHB_collate_onebatch,
                            shuffle=True,
                            drop_last=False)

    return train_loader, val_loader, restore_transform
