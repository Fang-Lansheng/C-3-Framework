import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision

from misc.networks.cnn import make_layers, cfg
from misc.networks.rnn import BiConvLSTM
from misc.networks.vision_transformer import VisionTransformer


class NewNet(nn.Module):
    def __init__(self, backbone='csrnet', patch_size=8, lstm_in_dim=512, num_sam_layers=6,
                 num_lstm_layers=1, load_weights=True, multi_fuse=True, is_lstm=False, is_vit=True):
        super(NewNet, self).__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.lstm_in_dim = lstm_in_dim
        self.num_sam_layers = num_sam_layers
        self.num_lstm_layers = num_lstm_layers
        self.load_weights = load_weights
        self.multi_fuse = multi_fuse
        self.is_lstm = is_lstm
        self.is_vit = is_vit

        self.features = make_layers(cfg[self.backbone])

        self.reg_layer = make_layers(cfg['csrnet-backend'], in_channels=512, dilation=2)

        self.density_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        if self.multi_fuse:
            scale_tree_list = []
            for i in range(self.num_sam_layers):
                if i == 0:
                    # module = ScaleTreeBlock(in_channels=512, out_channels=64)
                    module = ScaleTreeBlockRAW(in_channels=512, out_channels=64)
                else:
                    # module = ScaleTreeBlock(in_channels=64 * i, out_channels=64)
                    module = ScaleTreeBlockRAW(in_channels=64 * i, out_channels=64)
                scale_tree_list.append(module)
            self.multi_scale_module = nn.ModuleList(scale_tree_list)

            self.dense_fuse = nn.Sequential(
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU())

        if self.is_lstm:
            self.ascend_layer = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=self.lstm_in_dim, kernel_size=1),
                nn.ReLU())
            self.descend_layer = nn.Sequential(
                nn.Conv2d(in_channels=self.lstm_in_dim, out_channels=1, kernel_size=1),
                nn.ReLU())

            # self.encoder = TransformerEncoder()
            # self.decoder = TransformerDecoder()

            self.bi_conv_lstm = BiConvLSTM(input_size=(self.patch_size, self.patch_size),
                                           input_dim=self.lstm_in_dim, hidden_dim=self.lstm_in_dim,
                                           kernel_size=(3, 3), num_layers=self.num_lstm_layers)

        if self.is_vit:
            self.vit = VisionTransformer(patch_size=self.patch_size, in_channels=1,
                                         depth=8, num_heads=16, mlp_ratio=3.)

        if self.load_weights:
            if self.backbone == 'vgg-19':
                mod = torchvision.models.vgg19(pretrained=True)
                self._initialize_weights()
                self.features.load_state_dict(mod.features[:].state_dict())
            elif self.backbone == 'vgg-16':
                mod = torchvision.models.vgg16(pretrained=True)
                self._initialize_weights()
                self.features.load_state_dict(mod.features[:].state_dict())
            elif self.backbone == 'csrnet':
                mod = torchvision.models.vgg16(pretrained=True)
                self._initialize_weights()
                self.features.load_state_dict(mod.features[0:23].state_dict())
            elif self.backbone == 'amrnet':
                mod = torchvision.models.vgg19(pretrained=True)
                self._initialize_weights()
                self.features.load_state_dict(mod.features[0:27].state_dict())

    def forward(self, x):  # x.shape:  [B, 1, M / 8, N / 8] (Train: M = N = crop_size)
        # x, _ = self.get_density_map(x)
        # x = torch.nn.functional.upsample_bilinear(x, scale_factor=8)

        if self.is_lstm:
            # ~~~
            b, c, m, n = x.size()
            x, sort_idx, part = self._partition_and_sort(x, patch_size=self.patch_size)

            _, t, _, h, w = x.size()
            x = x.reshape([b, t, c * h * w])  # c = 1, h = w = patch_size

            # x = self.ascend_layer(x.reshape([b * t, c, h, w]))  # [b * t, c, h, w]
            # x = x.reshape([b, t, self.lstm_in_dim, h, w])  # [b, t, c, h, w]

            x = self.bi_conv_lstm(x)  # [b, t, 1, h, w]

            _, t, c, h, w = x.size()
            x = self.descend_layer(x.reshape([b * t, c, h, w]))
            x = x.reshape([b, t, 1, h, w])
            x = self._jagsaw(x, index=sort_idx, part=part)
            # ~~~
            #
            # b, c, h, w = x.size()
            # x_sum = x.view([b, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # x_norm = x / (x_sum + 1e-6)
            #
            # return x, x_norm  # [B, 1, M, N]

        if self.is_vit:
            x = self.vit(x)

        return x

    def get_density_map(self, imgs):  # img.shape:  [B, 3, M, N] (Train: M = N = crop_size)
        x = self.features(imgs)  # [B, 512, M / 8, N / 8]
        x = self.reg_layer(x)  # [B, 64, M / 8,  N / 8]

        if self.multi_fuse:
            for (i, module) in enumerate(self.multi_scale_module):
                if i < 1:
                    x = module(x, self.training)
                else:
                    y = module(x, self.training)
                    x = torch.cat((x, y), dim=1)
            x = self.dense_fuse(x)

        x = self.density_layer(x)  # [B,   1, M / 8,  N / 8]

        x = torch.nn.functional.upsample_bilinear(x, scale_factor=8)

        b, c, h, w = x.size()
        x_sum = x.view([b, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_norm = x / (x_sum + 1e-6)

        return x, x_norm

    def _forward_unimplemented(self, x) -> None:
        raise NotImplementedError

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _partition_and_sort(x, patch_size):
        # x = nn.functional.upsample_bilinear(density_maps, scale_factor=8)  # [B, C, M, N] (C=1)
        batch_size = x.shape[0]
        w, h = x.shape[-2:]
        m, n = int(w / patch_size), int(h / patch_size)

        # partition
        patches = []
        idx_list = []
        for b in range(batch_size):
            patches_batch = []
            patch_counts_batch = []
            for i in range(m):
                for j in range(n):
                    patch = x[b, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                    patch_count = patch.cpu().data.numpy().sum()
                    patch = patch.unsqueeze(0).unsqueeze(0)  # [1, 1, C, PATCH_SIZE, PATCH_SIZE]
                    patches_batch.append(patch)  # len = m * n, elem = torch.tensor
                    patch_counts_batch.append(patch_count)
            _, idx0 = torch.sort(torch.tensor(patch_counts_batch))
            _, idx = torch.sort(idx0)

            patches_batch = torch.cat(patches_batch, dim=1)
            patches_batch = patches_batch.index_select(dim=1, index=idx0.cuda())

            # # --------------------
            # # plot the patches
            # img_count_sum = 0
            # for i in range(m * n):
            #     img = patches_batch[0, i, 0, :, :].cpu().data.numpy()
            #     img_count = img.sum()
            #     img_count_sum += img_count
            #     img = cv2.resize(img, (256, 256))
            #     img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            #     img = (img * 255).astype(np.uint8)
            #     img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            #     cv2.imwrite('000_patch_{:d}_count_{:.2f}.png'.format(i, img_count), img)
            # print('STAGE 2 - count validation (partition): {:.4f}'.format(img_count_sum))
            # # --------------------

            idx_list.append(idx)

            patches.append(patches_batch)

        patches = torch.cat(patches, dim=0)

        # # patch_count
        # _, idx0 = torch.sort(torch.tensor(patch_counts))
        # _, idx = torch.sort(idx0)
        #
        # patches = torch.cat(patches, dim=1)  # [B, T, C, PATCH_SIZE, PATCH_SIZE] (T = m * n)
        # patches = patches.index_select(dim=1, index=idx0)

        return patches.cuda(), idx_list, [m, n]  # [B, T, C, PATCH_SIZE, PATCH_SIZE] (T = m * n)

    @staticmethod
    def _jagsaw(x, index, part):
        batch_size, _, c, patch_size, _ = x.shape
        imgs = []
        m, n = part
        for b in range(batch_size):
            patches = x[b]
            patches = torch.index_select(patches, dim=0, index=index[b].cuda())
            img = torch.zeros([1, c, patch_size * m, patch_size * n])
            for i in range(m):
                for j in range(n):
                    img[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = \
                        patches[i * n + j, :, :, :]
            imgs.append(img)
        output = torch.cat(imgs, dim=0)

        return output.cuda()


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

        x = self.max_out(x1, x2, x3, x4)
        return x

    def _forward_unimplemented(self, x) -> None:
        raise NotImplementedError

    @staticmethod
    def max_out(x1, x2, x3, x4):
        """ ~ x = max(x1, x2, x3, x4) """
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


class ScaleTreeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=256):
        super(ScaleTreeBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.root = nn.Conv2d(in_channels=self.in_channels, out_channels=hidden_dim, kernel_size=1)

        self.layer_1_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0, dilation=1)
        self.layer_1_2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.layer_1_3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=2)

        self.layer_2_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0, dilation=1)
        self.layer_2_2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.layer_2_3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=2)

        self.layer_2_4 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.layer_2_5 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=3, dilation=3)
        self.layer_2_6 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=4, dilation=4)

        self.layer_2_7 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=4, dilation=4)
        self.layer_2_8 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=5, dilation=5)
        self.layer_2_9 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=6, dilation=6)

        self.scale_fuse = nn.Sequential(
            # nn.Conv2d(in_channels=hidden_dim * 9, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=hidden_dim * 9, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, is_training=True):
        f = self.root(x)

        f1 = self.layer_1_1(f)
        f2 = self.layer_1_2(f)
        f3 = self.layer_1_3(f)

        if is_training:
            alpha, beta = torch.rand(1).cuda(), torch.rand(1).cuda()
        else:
            alpha, beta = torch.tensor(0.5).cuda(), torch.tensor(0.5).cuda()

        f1_hat = f1
        f2_hat = alpha * f1_hat + (1 - alpha) * f2
        f3_hat = beta * f2_hat + (1 - beta) * f3

        s1 = self.layer_2_1(f1_hat)
        s2 = self.layer_2_2(f1_hat)
        s3 = self.layer_2_3(f1_hat)

        s4 = self.layer_2_4(f2_hat)
        s5 = self.layer_2_5(f2_hat)
        s6 = self.layer_2_6(f2_hat)

        s7 = self.layer_2_7(f3_hat)
        s8 = self.layer_2_8(f3_hat)
        s9 = self.layer_2_9(f3_hat)

        if is_training:
            alpha_1, beta_1 = torch.rand(1).cuda(), torch.rand(1).cuda()
            alpha_2, beta_2 = torch.rand(1).cuda(), torch.rand(1).cuda()
            alpha_3, beta_3 = torch.rand(1).cuda(), torch.rand(1).cuda()
        else:
            alpha_1, beta_1 = torch.tensor(0.5).cuda(), torch.tensor(0.5).cuda()
            alpha_2, beta_2 = torch.tensor(0.5).cuda(), torch.tensor(0.5).cuda()
            alpha_3, beta_3 = torch.tensor(0.5).cuda(), torch.tensor(0.5).cuda()
        # alpha_1, alpha_2, alpha_3 = alpha, alpha, alpha
        # beta_1, beta_2, beta_3 = beta, beta, beta

        s1_hat = s1
        s2_hat = alpha_1 * s1_hat + (1 - alpha_1) * s2
        s3_hat = beta_1 * s2_hat + (1 - beta_1) * s3

        s4_hat = s4
        s5_hat = alpha_2 * s4_hat + (1 - alpha_2) * s5
        s6_hat = beta_2 * s5_hat + (1 - beta_2) * s6

        s7_hat = s7
        s8_hat = alpha_3 * s7_hat + (1 - alpha_3) * s8
        s9_hat = beta_3 * s8_hat + (1 - beta_3) * s9

        output = self.scale_fuse(torch.cat((s1_hat, s2_hat, s3_hat, s4_hat, s5_hat,
                                            s6_hat, s7_hat, s8_hat, s9_hat), dim=1))

        return output

    def _forward_unimplemented(self, x) -> None:
        raise NotImplementedError


class ScaleTreeBlockRAW(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=256):
        super(ScaleTreeBlockRAW, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.root = nn.Conv2d(in_channels=self.in_channels, out_channels=hidden_dim, kernel_size=1)

        self.layer_1_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0, dilation=1)
        self.layer_1_2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.layer_1_3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=6, dilation=3)

        self.layer_2_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0, dilation=1)
        self.layer_2_2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.layer_2_3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=6, dilation=3)

        self.layer_2_4 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=6, dilation=3)
        self.layer_2_5 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding=12, dilation=4)
        self.layer_2_6 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=9, padding=20, dilation=5)

        self.layer_2_7 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=9, padding=20, dilation=5)
        self.layer_2_8 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=11, padding=30, dilation=6)
        self.layer_2_9 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=13, padding=42, dilation=7)

        self.scale_fuse = nn.Sequential(
            # nn.Conv2d(in_channels=hidden_dim * 9, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=hidden_dim * 9, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, is_training=True):
        f = self.root(x)

        f1 = self.layer_1_1(f)
        f2 = self.layer_1_2(f)
        f3 = self.layer_1_3(f)

        if is_training:
            alpha, beta = torch.rand(1).cuda(), torch.rand(1).cuda()
        else:
            alpha, beta = torch.tensor(0.5).cuda(), torch.tensor(0.5).cuda()

        f1_hat = f1
        f2_hat = alpha * f1_hat + (1 - alpha) * f2
        f3_hat = beta * f2_hat + (1 - beta) * f3

        s1 = self.layer_2_1(f1_hat)
        s2 = self.layer_2_2(f1_hat)
        s3 = self.layer_2_3(f1_hat)

        s4 = self.layer_2_4(f2_hat)
        s5 = self.layer_2_5(f2_hat)
        s6 = self.layer_2_6(f2_hat)

        s7 = self.layer_2_7(f3_hat)
        s8 = self.layer_2_8(f3_hat)
        s9 = self.layer_2_9(f3_hat)

        if is_training:
            alpha_1, beta_1 = torch.rand(1).cuda(), torch.rand(1).cuda()
            alpha_2, beta_2 = torch.rand(1).cuda(), torch.rand(1).cuda()
            alpha_3, beta_3 = torch.rand(1).cuda(), torch.rand(1).cuda()
        else:
            alpha_1, beta_1 = torch.tensor(0.5).cuda(), torch.tensor(0.5).cuda()
            alpha_2, beta_2 = torch.tensor(0.5).cuda(), torch.tensor(0.5).cuda()
            alpha_3, beta_3 = torch.tensor(0.5).cuda(), torch.tensor(0.5).cuda()
        # alpha_1, alpha_2, alpha_3 = alpha, alpha, alpha
        # beta_1, beta_2, beta_3 = beta, beta, beta

        s1_hat = s1
        s2_hat = alpha_1 * s1_hat + (1 - alpha_1) * s2
        s3_hat = beta_1 * s2_hat + (1 - beta_1) * s3

        s4_hat = s4
        s5_hat = alpha_2 * s4_hat + (1 - alpha_2) * s5
        s6_hat = beta_2 * s5_hat + (1 - beta_2) * s6

        s7_hat = s7
        s8_hat = alpha_3 * s7_hat + (1 - alpha_3) * s8
        s9_hat = beta_3 * s8_hat + (1 - beta_3) * s9

        output = self.scale_fuse(torch.cat((s1_hat, s2_hat, s3_hat, s4_hat, s5_hat,
                                            s6_hat, s7_hat, s8_hat, s9_hat), dim=1))

        return output

    def _forward_unimplemented(self, x) -> None:
        raise NotImplementedError
