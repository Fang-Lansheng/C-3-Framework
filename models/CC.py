from misc.utils import *
from misc.losses.ssim_loss import SSIM_Loss


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()

        self.model_name = model_name
        net = None

        if self.model_name == 'AlexNet':
            from .SCC_Model.AlexNet import AlexNet as net
        elif self.model_name == 'VGG':
            from .SCC_Model.VGG import VGG as net
        elif self.model_name == 'VGG_DECODER':
            from .SCC_Model.VGG_decoder import VGG_decoder as net
        elif self.model_name == 'MCNN':
            from .SCC_Model.MCNN import MCNN as net
        elif self.model_name == 'CSRNet':
            from .SCC_Model.CSRNet import CSRNet as net
        elif self.model_name == 'Res50':
            from .SCC_Model.Res50 import Res50 as net
        elif self.model_name == 'Res101':
            from .SCC_Model.Res101 import Res101 as net
        elif self.model_name == 'Res101_SFCN':
            from .SCC_Model.Res101_SFCN import Res101_SFCN as net
        elif self.model_name == 'LibraNet':
            from .SCC_Model.LibraNet import LibraNet as net
        elif self.model_name == 'NewNet':
            from .SCC_Model.NewNet import NewNet as net

        self.CCN = net()

        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()

        # loss functions
        self.lambda1 = 1e-5
        self.loss_fn_mse = nn.MSELoss().cuda()
        self.loss_fn_count = nn.L1Loss().cuda()
        self.loss_fn_ssim = SSIM_Loss(in_channels=1).cuda()
        # losses
        self.losses = None
        self.loss_mse = None
        self.loss_count = None
        self.loss_ssim = None

    @property
    def loss(self):
        return self.losses

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        self.losses = self.build_loss(density_map.squeeze(), gt_map.squeeze())
        return density_map

    def build_loss(self, density_map, gt_data):
        self.loss_mse = self.loss_fn_mse(density_map, gt_data)
        # self.count_loss = self.fn_count_loss(density_map.sum(), gt_data.sum())
        # return self.mse_loss + self.lambda1 * self.count_loss
        return self.loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
