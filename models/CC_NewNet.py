from misc.utils import *
from misc.losses.ssim_loss import SSIM_Loss
from misc.losses.lcm_loss import LCM_Loss
from misc.losses.ot_loss import OT_Loss
from config import cfg


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()

        self.model_name = model_name
        net = None

        if self.model_name == 'NewNet':
            from .SCC_Model.NewNet import NewNet as net

        self.CCN = net(multi_fuse=cfg.NET_MULTI_FUSE,
                       is_lstm=cfg.NET_IS_LSTM,
                       is_vit=cfg.NET_IS_VIT)
        print(self.CCN)

        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()

        # loss functions
        self.lambda1 = 1e-2
        self.loss_fn_mse = nn.MSELoss().cuda()
        self.loss_fn_count = nn.L1Loss().cuda()
        self.loss_fn_ssim = SSIM_Loss(in_channels=1).cuda()
        self.loss_fn_lcm = LCM_Loss().cuda()
        # self.loss_fn_ot = OT_Loss(c_size=256, stride=8, norm_cood=0, device=self.cuda())
        # losses
        self.losses = None
        self.loss_mse = None
        self.loss_count = None
        self.loss_ssim = None
        self.loss_lcm = None

    @property
    def loss(self):
        return self.losses

    def forward(self, img, gt_map):  # img.shape: [B, C=3, 256, 256], gt_map.shape: [B, 32, 32]
        # dm, _ = self.CCN.get_density_map(img)  # dm.shape: [B, C=1, 32, 32]
        # dm_x8 = torch.nn.functional.upsample_bilinear(dm, scale_factor=8)
        # self.loss_mse = self.loss_fn_mse(dm_x8.squeeze(), gt_map)
        #
        # lcm = self.CCN(dm)
        # self.loss_lcm = self.loss_fn_lcm(lcm, gt_map)
        #
        # self.losses = self.loss_mse + self.loss_lcm * self.lambda1
        #
        # return lcm
        dm, _ = self.CCN.get_density_map(img)
        output = self.CCN(dm)
        self.loss_mse = self.loss_fn_mse(output.squeeze(1), gt_map)
        self.losses = self.loss_mse

        return output

    def build_loss(self, density_map, gt_data):
        self.loss_mse = self.loss_fn_mse(density_map, gt_data)
        # self.count_loss = self.fn_count_loss(density_map.sum(), gt_data.sum())
        # return self.mse_loss + self.lambda1 * self.count_loss
        return self.loss_mse

    def test_forward(self, img):
        dm, _ = self.CCN.get_density_map(img)
        output = self.CCN(dm)
        return output

