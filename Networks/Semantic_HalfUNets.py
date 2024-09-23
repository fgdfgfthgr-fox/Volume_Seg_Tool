import torch
import torch.nn as nn
from .Modules.HalfUNets_Components import GhostDoubleConv, merge
from .Modules.General_Components import ResBasicBlock, ResBottleneckBlock, BasicBlock, scSE
import math


# Lu H, She Y, Tie J, et al. Half-UNet: A simplified U-Net architecture for medical image segmentation[J]. Frontiers
# in Neuroinformatics, 2022, 16: 911679.

# Similar to UNet, but with a simplified decoder structure.
# Basic: Use simple double conv block
# Ghost: The original design from the Half-UNet paper, which uses the ghost module instead of normal conv, take fewer parameters but not faster
# Residual: Use residual block
# ResidualBottleneck: Use residual bottleneck block

# Don't have an instance segmentation version.


class HalfUNet(nn.Module):

    def __init__(self, base_channels=16, depth=5, z_to_xy_ratio=1, type='Basic', se=False, unsupervised=False, label_mean=torch.tensor(0.5)):
        super(HalfUNet, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        self.se = se
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = (3, 3, 3)
        block = {'Basic': BasicBlock, 'Ghost': GhostDoubleConv, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        kernel_sizes_down = [(1, 4, 4) if self.special_layers > 0 and i < self.special_layers else
                             (4, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (4, 4, 4) for i in range(depth)]
        padding_down = [(0, 1, 1) if self.special_layers > 0 and i < self.special_layers else
                        (1, 0, 0) if self.special_layers < 0 and i < -self.special_layers else
                        (1, 1, 1) for i in range(depth)]
        stride_down = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                       (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                       (2, 2, 2) for i in range(depth)]

        for i in range(depth):
            if i == 0:
                setattr(self, f'encode{i}',
                        BasicBlock(1, base_channels, kernel_sizes_conv, 1))
            else:
                setattr(self, f'encode{i}',
                        block(base_channels, base_channels, kernel_sizes_conv))
            if se: setattr(self, f'encode_se{i}', scSE(base_channels))
            if i != depth - 1:
                setattr(self, f'down{i}',
                        nn.Conv3d(base_channels, base_channels, kernel_sizes_down[i], stride_down[i], padding_down[i]))

        self.decoder = block(base_channels, base_channels, kernel_sizes_conv, 1)
        if se: self.decoder_se = scSE(base_channels)
        if unsupervised:
            self.u_decoder = block(base_channels, base_channels, kernel_sizes_conv, 1)
            if se: self.u_decoder_se = scSE(base_channels)
        logit_label_mean = torch.log(label_mean / (1 - label_mean)) * 0.5
        self.s_out = nn.Conv3d(base_channels, 1, kernel_size=1)
        with torch.no_grad():
            self.s_out.bias.fill_(logit_label_mean)
        if unsupervised:
            self.u_out = nn.Conv3d(base_channels, 1, kernel_size=1)

    def semantic_decode(self, x):
        x = self.decoder(x)
        if self.se: x = self.decoder_se(x)
        x = self.s_out(x)
        return x

    def unsupervised_decode(self, x):
        u_x = self.u_decoder(x)
        if self.se: u_x = self.u_decoder_se(u_x)
        u_x = self.u_out(u_x)
        return u_x

    def forward(self, x, type=(1,)):
        _, _, D, H, W = x.shape
        for i in range(0, self.depth):
            x = getattr(self, f'encode{i}')(x)
            if self.se: x = getattr(self, f"encode_se{i}")(x)
            if i == 0:
                x_out = x
            else:
                x_out = x_out + torch.nn.functional.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=True)
            if i != self.depth - 1:
                x = getattr(self, f"down{i}")(x)

        if type[0] == 0:
            return self.semantic_decode(x_out)
        elif type[0] == 1:
            return self.unsupervised_decode(x_out)
        elif type[0] == 2:
            return self.semantic_decode(x_out), self.unsupervised_decode(x_out)