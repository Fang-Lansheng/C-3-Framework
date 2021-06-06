import torch
import torch.nn as nn


class LCM_Loss(nn.Module):
    def __init__(self, input_size=32, patch_size=8, resize=None):
        super(LCM_Loss, self).__init__()
        assert input_size % patch_size == 0

        self.c_size = input_size  # default: 32
        self.kernel_size = patch_size  # default: 8
        self.resize = resize
        self.loss_func = nn.L1Loss().cuda()

        self.weight = torch.ones(1, 1, 8 * self.kernel_size, 8 * self.kernel_size,
                                 requires_grad=False).cuda()
        self.filter = nn.functional.conv2d

    def forward(self, est_map, gt_map):  # est_map.shape: [B, 1, 16], gt_map.shape: [B, 32, 32]
        gt_lcm = self.filter(gt_map.unsqueeze(1), self.weight, stride=8 * self.kernel_size)
        loss = self.loss_func(est_map, gt_lcm.reshape(gt_lcm.shape[0], gt_lcm.shape[1], -1))

        return loss

    def _forward_unimplemented(self, x) -> None:
        raise NotImplementedError
