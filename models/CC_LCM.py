import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from config import cfg
from misc.utils import *

if cfg.DATASET == 'SHHB':
    from datasets.SHHB.setting import cfg_data
elif cfg.DATASET == 'SHHA':
    from datasets.SHHA.setting import cfg_data
elif cfg.DATASET == 'UCSD':
    from datasets.UCSD.setting import cfg_data
elif cfg.DATASET == 'Mall':
    from datasets.Mall.setting import cfg_data
elif cfg.DATASET == 'FDST':
    from datasets.FDST.setting import cfg_data

class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name, pretrained=True):
        super(CrowdCounter, self).__init__()

        self.model_name = model_name
        net = None

        if model_name == 'AMRNet':
            from .SCC_Model.AMRNet import AMRNet as net

        self.CCN = net(pretrained)

        if len(gpus) > 1:  # for multi gpu
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda(gpus[0])
        else:  # for one gpu
            self.CCN = self.CCN.cuda()

        self.loss_sum_fn = nn.L1Loss().cuda()

        self.SumLoss = True

    @property
    def loss(self):
        return self.loss_total

    def loss_sum(self):
        return self.loss_sum

    def forward(self, img, gt_map):
        count_map = self.CCN(img)
        gt_map = torch.unsqueeze(gt_map, 1)
        self.loss_total, self.loss_sum = self.build_loss(count_map, gt_map)

        return count_map

    def build_loss(self, count_map, gt_map):
        loss_total, loss_sum_all = 0., 0.

        if self.SumLoss:
            gt_map_ = gt_map / cfg_data.LOG_PARA
            kernel3, kernel4, kernel5 = 2, 4, 8

            # filter3 = torch.ones(1, 1, kernel3, kernel3, requires_grad=False).cuda()
            # filter4 = torch.ones(1, 1, kernel4, kernel4, requires_grad=False).cuda()
            filter5 = torch.ones(1, 1, kernel5, kernel5, requires_grad=False).cuda()

            # gt_lcm_3 = F.conv2d(gt_map_, filter3, stride=kernel3)
            # gt_lcm_4 = F.conv2d(gt_map_, filter4, stride=kernel4)
            gt_lcm_5 = F.conv2d(gt_map_, filter5, stride=kernel5)

            loss_sum_all = self.loss_sum_fn(count_map, gt_lcm_5)
            loss_total += loss_sum_all

        return loss_total, loss_sum_all

    def test_forward(self, img):
        count_map = self.CCN(img)
        return count_map
