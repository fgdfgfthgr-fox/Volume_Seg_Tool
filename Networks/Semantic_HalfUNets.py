import torch.nn as nn
from .Modules.HalfUNets_Components import GhostDoubleConv, merge
from .Modules.General_Components import BasicBlock, ResBasicBlock, ResBottleneckBlock
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

    def __init__(self, base_channels=16, depth=5, z_to_xy_ratio=1, type='Basic'):
        super(HalfUNet, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]
        block = {'Basic': BasicBlock, 'Ghost': GhostDoubleConv, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]

        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        block(1, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'decoder',
                        block(base_channels, base_channels, kernel_sizes_conv[0]))
            else:
                setattr(self, f'encode{i}',
                        block(base_channels, base_channels, kernel_sizes_conv[i]))

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = getattr(self, f'encode0')(x)
        x_1 = x
        for i in range(1, self.depth):
            x_1 = self.max_pool_flat(x_1) if self.special_layers > 0 and i < self.special_layers else \
                self.max_pool_shrink(x_1) if self.special_layers < 0 and i < -self.special_layers else \
                self.max_pool(x_1)
            x_1 = getattr(self, f"encode{i}")(x_1)
            x = merge(x, x_1)
        x = self.out(x)
        return x
