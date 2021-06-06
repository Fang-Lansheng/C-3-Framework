import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
from config import cfg

patch_max = cfg.PATCHMAX[cfg.DATASET]


class AMRNet(nn.Module):
    def __init__(self, load_weights=False, stage_num=[3, 3, 3], count_range=patch_max, lambda_i=1., lambda_k=1.):
        super(AMRNet, self).__init__()

        # cfg
        self.stage_num = stage_num
        self.lambda_i = lambda_i  # ~~~ lambda for shifting factor (beta)
        self.lambda_k = lambda_k  # ~~~ lambda for scaling factor (gamma)
        self.count_range = count_range
        self.multi_fuse = True
        self.soft_interval = True

        self.layer3 = self.VGG_make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512])
        self.layer4 = self.VGG_make_layers(['M', 512, 512, 512], in_channels=512)
        self.layer5 = self.VGG_make_layers(['M', 512, 512, 512], in_channels=512)

        if self.multi_fuse:  # ~ True
            self.fuse_layer5 = DC_layer(level=0)
            self.fuse_layer4 = DC_layer(level=1)
            self.fuse_layer3 = DC_layer(level=2)

        self.count_layer5 = Count_layer(pool=2)
        self.count_layer4 = Count_layer(pool=4)
        self.count_layer3 = Count_layer(pool=8)

        if self.soft_interval:
            self.layer5_k = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.Tanh(),
            )
            self.layer4_k = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.Tanh(),
            )
            self.layer3_k = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.Tanh(),
            )

            self.layer5_i = nn.Sequential(
                nn.Conv2d(512, self.stage_num[0], kernel_size=1),
                nn.Sigmoid(),
            )
            self.layer4_i = nn.Sequential(
                nn.Conv2d(512, self.stage_num[1], kernel_size=1),
                nn.Sigmoid(),
            )
            self.layer3_i = nn.Sequential(
                nn.Conv2d(512, self.stage_num[2], kernel_size=1),
                nn.Sigmoid(),
            )

        self.layer5_p = nn.Sequential(
            nn.Conv2d(512, self.stage_num[0], kernel_size=1),
            nn.ReLU(),
        )
        self.layer4_p = nn.Sequential(
            nn.Conv2d(512, self.stage_num[1], kernel_size=1),
            nn.ReLU(),
        )
        self.layer3_p = nn.Sequential(
            nn.Conv2d(512, self.stage_num[2], kernel_size=1),
            nn.ReLU(),
        )

        if load_weights:
            # self._initialize_weights()

            mod = models.vgg16(pretrained=True)
            # pretrain_path = './models/Pretrain_Model/vgg16-397923af.pth'
            # mod.load_state_dict(torch.load(pretrain_path))

            pretrain_mode_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
            from torch.utils.model_zoo import load_url
            state_dict = load_url(pretrain_mode_url)
            mod.load_state_dict(state_dict)

            new_state_dict = OrderedDict()
            for key, params in mod.features[0:23].state_dict().items():
                new_state_dict[key] = params
            self.layer3.load_state_dict(new_state_dict)

            new_state_dict = OrderedDict()
            for key, params in mod.features[23:30].state_dict().items():
                key = str(int(key[:2]) - 23) + key[2:]
                new_state_dict[key] = params
            self.layer4.load_state_dict(new_state_dict)

            new_state_dict = OrderedDict()
            for key, params in mod.features[23:30].state_dict().items():
                key = str(int(key[:2]) - 23) + key[2:]
                new_state_dict[key] = params
            self.layer5.load_state_dict(new_state_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ~ VGG-16 backbone
        # ~ x.shape: [1, 3, M, N]
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        # ~ Scale Aware Module
        if self.multi_fuse:  # ~ Ture
            x5 = self.fuse_layer5(x5)  # ~ x5.shape: [1, 512, M / 32, N / 32]
            x4 = self.fuse_layer4(x4)  # ~ x4.shape: [1, 512, M / 16, N / 16]
            x3 = self.fuse_layer3(x3)  # ~ x3.shape: [1, 512, M / 8,  N / 8]

        # ~ Mixture Regression Module (MRM) - layer 5
        x5_ = self.count_layer5(x5)  # ~ x5_.shape [1, 512, M / 64, N / 64]
        # ~ Adaptive Soft Interval Module (ASIM)
        p5 = self.layer5_p(x5_)  # ~ probability vector factor | p5.shape: [1, 3, M / 64, N / 64]
        if self.soft_interval:  # ~ True
            k5 = self.layer5_k(x5_)  # ~ scaling factor            | k5.shape: [1, 1, M / 64, N / 64]
            i5 = self.layer5_i(x5_)  # ~ shifting vector factor    | i5.shape: [1, 3, M / 64, N / 64]

        # ~ Mixture Regression Module (MRM) - layer 4
        x4_ = self.count_layer4(x4)  # ~ x4_.shape [1, 512, M / 64, N / 64]
        # ~ Adaptive Soft Interval Module (ASIM)
        p4 = self.layer4_p(x4_)  # ~ probability vector factor | p4.shape: [1, 3, M / 64, N / 64]
        if self.soft_interval:  # ~ True
            k4 = self.layer4_k(x4_)  # ~ scaling factor            | k4.shape: [1, 1, M / 64, N / 64]
            i4 = self.layer4_i(x4_)  # ~ shifting vector factor    | i4.shape: [1, 3, M / 64, N / 64]

        # ~ Mixture Regression Module (MRM) - layer 3
        x3_ = self.count_layer3(x3)  # ~ x3_.shape [1, 512, M / 64, N / 64]
        # ~ Adaptive Soft Interval Module (ASIM)
        p3 = self.layer3_p(x3_)  # ~ probability vector factor | p3.shape: [1, 3, M / 64, N / 64]
        if self.soft_interval:  # ~ True
            k3 = self.layer3_k(x3_)  # ~ scaling factor            | k3.shape: [1, 1, M / 64, N / 64]
            i3 = self.layer3_i(x3_)  # ~ shifting vector factor    | i3.shape: [1, 3, M / 64, N / 64]

        stage1_regress = p5[:, 0, :, :] * 0  # ~ stage1_regress.shape: [1, M / 64, N / 64]
        stage2_regress = p4[:, 0, :, :] * 0  # ~ stage2_regress.shape: [1, M / 64, N / 64]
        stage3_regress = p3[:, 0, :, :] * 0  # ~ stage3_regress.shape: [1, M / 64, N / 64]

        # ~ self.stage_num = s = [s_1, s_2, s_3]
        for index in range(self.stage_num[0]):  # ~ i (index): 0, 1, 2; k = 1; s_k (= s_1 = 3)
            if self.soft_interval:  # True
                # ~ lambda_i (= 1.0): lambda for shifting factor (beta)
                # ~ i5 ([1, 3, M / 64, N / 64]): shifting vector factor
                # ~ p5 ([1, 3, M / 64, N / 64]): probability vector factor
                stage1_regress = stage1_regress + (float(index) + self.lambda_i * i5[:, index, :, :]) * p5[:, index, :,
                                                                                                        :]
                # ~ k=1, \sum_{i=1}^{s_k} p^{(k)}_{i} \cdot (i + \beta^{(k)}_{i})
            else:
                stage1_regress = stage1_regress + float(index) * p5[:, index, :, :]
                # ~ k=1, \sum_{k=1}^{s_k} p^{(k)}_{i} \cdot i
        stage1_regress = torch.unsqueeze(stage1_regress, 1)  # ~ stage1_regress.shape: [1, 1, M / 64, N / 64]
        if self.soft_interval:  # True
            # ~ lambda_k (= 1.0): lambda for scaling factor (gamma)
            # ~ k5 ([1, 1, M / 64, M / 64]: scaling factor
            stage1_regress = stage1_regress / (float(self.stage_num[0]) * (1. + self.lambda_k * k5))
            # ~ k=1, \sum_{k=1}^{s_k} \frac{p^{(k)}_{i} \cdot i}{s_1(1+\gamma^{(k)})}
        else:
            stage1_regress = stage1_regress / float(self.stage_num[0])
            # ~ k=1, \sum_{k=1}^{s_k} \frac{p^{(k)}_{i} \cdot i}{s_1}

        for index in range(self.stage_num[1]):  # ~ i (index): 0, 1, 2; k = 2; s_k (= s_2 = 3)
            if self.soft_interval:
                # ~ lambda_i (= 1.0): lambda for shifting factor (beta)
                # ~ i4 ([1, 3, M / 64, N / 64]): shifting vector factor
                # ~ p4 ([1, 3, M / 64, N / 64]): probability vector factor
                stage2_regress = stage2_regress + (float(index) + self.lambda_i * i4[:, index, :, :]) * p4[:, index, :,
                                                                                                        :]
                # ~ k=2, \sum_{i=1}^{s_k} p^{(k)}_{i} \cdot (i + \beta^{(k)}_{i})
            else:
                stage2_regress = stage2_regress + float(index) * p4[:, index, :, :]
                # ~ k=2, \sum_{k=1}^{s_k} p^{(k)}_{i} \cdot i
        stage2_regress = torch.unsqueeze(stage2_regress, 1)
        if self.soft_interval:
            # ~ lambda_k (= 1.0): lambda for scaling factor (gamma)
            # ~ k4 ([1, 1, M / 64, M / 64]
            stage2_regress = stage2_regress / ((float(self.stage_num[0]) * (1. + self.lambda_k * k5)) *
                                               (float(self.stage_num[1]) * (1. + self.lambda_k * k4)))
            # ~ k=2, \sum_{k=1}^{s_k} \frac{p^{(k)}_{i} \cdot (i + \beta^{(k)}_{i})}{\prod_{j=1}^k [s_j(1+\gamma^{(j)})]}
        else:
            stage2_regress = stage2_regress / float(self.stage_num[0] * self.stage_num[1])
            # ~ k=2, \sum_{k=1}^{s_k} \frac{p^{(k)}_{i} \cdot i}{\prod_{j=1}^ks_j}

        for index in range(self.stage_num[2]):  # ~ i (index): 0, 1, 2; k = 3; s_k (= s_3 = 3)
            if self.soft_interval:
                # ~ lambda_i (= 1.0): lambda for shifting factor (beta)
                # ~ i3 ([1, 3, M / 64, N / 64]): shifting vector factor
                # ~ p3 ([1, 3, M / 64, N / 64]): probability vector factor
                stage3_regress = stage3_regress + (float(index) + self.lambda_i * i3[:, index, :, :]) * p3[:, index, :,
                                                                                                        :]
                # ~ k=3, \sum_{i=1}^{s_k} p^{(k)}_{i} \cdot (i + \beta^{(k)}_{i})
            else:
                stage3_regress = stage3_regress + float(index) * p3[:, index, :, :]
                # ~ k=3, \sum_{k=1}^{s_k} p^{(k)}_{i} \cdot i
        stage3_regress = torch.unsqueeze(stage3_regress, 1)
        if self.soft_interval:
            stage3_regress = stage3_regress / ((float(self.stage_num[0]) * (1. + self.lambda_k * k5)) *
                                               (float(self.stage_num[1]) * (1. + self.lambda_k * k4)) *
                                               (float(self.stage_num[2]) * (1. + self.lambda_k * k3)))
            # ~ k=3, \sum_{k=1}^{s_k} \frac{p^{(k)}_{i} \cdot (i + \beta^{(k)}_{i})}{\prod_{j=1}^k [s_j(1+\gamma^{(j)})]}
        else:
            stage3_regress = stage3_regress / float(self.stage_num[0] * self.stage_num[1] * self.stage_num[2])
            # ~ k=3, \sum_{k=1}^{s_k} \frac{p^{(k)}_{i} \cdot i}{\prod_{j=1}^ks_j}

        # regress_count = stage1_regress * self.count_range
        # regress_count = (stage1_regress + stage2_regress) * self.count_range
        regress_count = (stage1_regress + stage2_regress + stage3_regress) * self.count_range
        # ~ C_p = C_m \sum_{k=1}^K \cdots

        return regress_count

    def VGG_make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=1):
        d_rate = dilation
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class Count_layer(nn.Module):
    """ ~ counting the LCM """

    def __init__(self, inplanes=512, pool=2):
        super(Count_layer, self).__init__()
        self.avgpool_layer = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((pool, pool), stride=pool),
        )
        self.maxpool_layer = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((pool, pool), stride=pool),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inplanes * 2, inplanes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_avg = self.avgpool_layer(x)
        x_max = self.maxpool_layer(x)

        x = torch.cat([x_avg, x_max], dim=1)
        x = self.conv1x1(x)
        return x


class DC_layer(nn.Module):
    def __init__(self, level, fuse=False):
        super(DC_layer, self).__init__()
        self.level = level  # ~ `level`: just a variable
        self.conv1x1_d1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_d2 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_d3 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_d4 = nn.Conv2d(512, 512, kernel_size=1)

        self.conv_d1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv_d3 = nn.Conv2d(512, 512, kernel_size=3, padding=3, dilation=3)
        self.conv_d4 = nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4)

        self.fuse = fuse
        if self.fuse:
            self.fuse = nn.Conv2d(512 * 2, 512, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1x1_d1(x)
        x2 = self.conv1x1_d2(x)
        x3 = self.conv1x1_d3(x)
        x4 = self.conv1x1_d4(x)

        x1 = self.conv_d1(x1)
        x2 = self.conv_d2(x2)
        x3 = self.conv_d3(x3)
        x4 = self.conv_d4(x4)

        # x = torch.cat([x1, x2, x3, x4], dim=1)
        # x = self.relu(self.fuse(x))
        x = Maxout(x1, x2, x3, x4)
        return x


def Maxout(x1, x2, x3, x4):
    """ ~ x = max(x1, x2, x3, x4)"""
    mask_1 = torch.ge(x1, x2)  # ~ computes x1 >= x2 element-wise
    mask_1 = mask_1.float()
    x = mask_1 * x1 + (1 - mask_1) * x2

    mask_2 = torch.ge(x, x3)
    mask_2 = mask_2.float()
    x = mask_2 * x + (1 - mask_2) * x3

    mask_3 = torch.ge(x, x4)
    mask_3 = mask_3.float()
    x = mask_3 * x + (1 - mask_3) * x4
    return x
