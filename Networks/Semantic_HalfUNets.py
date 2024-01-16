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


class Basic(nn.Module):

    def __init__(self, base_channels=16, depth=5, z_to_xy_ratio=1):
        super(Basic, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = []
        for i in range(depth):
            if self.special_layers > 0 and i < self.special_layers:
                kernel_sizes_conv.append((1, 3, 3))
            elif self.special_layers < 0 and i < -self.special_layers:
                kernel_sizes_conv.append((3, 1, 1))
            else:
                kernel_sizes_conv.append((3, 3, 3))

        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        BasicBlock(1, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'decoder',
                        BasicBlock(base_channels, base_channels, kernel_sizes_conv[0]))
            else:
                setattr(self, f'encode{i}',
                        BasicBlock(base_channels, base_channels, kernel_sizes_conv[i]))

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = getattr(self, f'encode0')(x)
        x_1 = x
        for i in range(1, self.depth):
            if self.special_layers > 0 and i < self.special_layers:
                x_1 = self.max_pool_flat(x_1)
            elif self.special_layers < 0 and i < -self.special_layers:
                x_1 = self.max_pool_shrink(x_1)
            else:
                x_1 = self.max_pool(x_1)
            x_1 = getattr(self, f"encode{i}")(x_1)
            x = merge(x, x_1)
        x = self.out(x)
        return x


class Ghost(nn.Module):

    def __init__(self, base_channels=16, depth=5, z_to_xy_ratio=1):
        super(Ghost, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = []
        for i in range(depth):
            if self.special_layers > 0 and i < self.special_layers:
                kernel_sizes_conv.append((1, 3, 3))
            elif self.special_layers < 0 and i < -self.special_layers:
                kernel_sizes_conv.append((3, 1, 1))
            else:
                kernel_sizes_conv.append((3, 3, 3))

        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        GhostDoubleConv(1, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'decoder',
                        GhostDoubleConv(base_channels, base_channels, kernel_sizes_conv[0]))
            else:
                setattr(self, f'encode{i}',
                        GhostDoubleConv(base_channels, base_channels, kernel_sizes_conv[i]))

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = getattr(self, f'encode0')(x)
        x_1 = x
        for i in range(1, self.depth):
            if self.special_layers > 0 and i < self.special_layers:
                x_1 = self.max_pool_flat(x_1)
            elif self.special_layers < 0 and i < -self.special_layers:
                x_1 = self.max_pool_shrink(x_1)
            else:
                x_1 = self.max_pool(x_1)
            x_1 = getattr(self, f"encode{i}")(x_1)
            x = merge(x, x_1)
        x = self.out(x)
        return x


class Residual(nn.Module):

    def __init__(self, base_channels=16, depth=5, z_to_xy_ratio=1):
        super(Residual, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = []
        for i in range(depth):
            if self.special_layers > 0 and i < self.special_layers:
                kernel_sizes_conv.append((1, 3, 3))
            elif self.special_layers < 0 and i < -self.special_layers:
                kernel_sizes_conv.append((3, 1, 1))
            else:
                kernel_sizes_conv.append((3, 3, 3))

        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        ResBasicBlock(1, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'decoder',
                        ResBasicBlock(base_channels, base_channels, kernel_sizes_conv[0]))
            else:
                setattr(self, f'encode{i}',
                        ResBasicBlock(base_channels, base_channels, kernel_sizes_conv[i]))

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = getattr(self, f'encode0')(x)
        x_1 = x
        for i in range(1, self.depth):
            if self.special_layers > 0 and i < self.special_layers:
                x_1 = self.max_pool_flat(x_1)
            elif self.special_layers < 0 and i < -self.special_layers:
                x_1 = self.max_pool_shrink(x_1)
            else:
                x_1 = self.max_pool(x_1)
            x_1 = getattr(self, f"encode{i}")(x_1)
            x = merge(x, x_1)
        x = self.out(x)
        return x


class ResidualBottleneck(nn.Module):

    def __init__(self, base_channels=16, depth=5, z_to_xy_ratio=1):
        super(ResidualBottleneck, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = []
        for i in range(depth):
            if self.special_layers > 0 and i < self.special_layers:
                kernel_sizes_conv.append((1, 3, 3))
            elif self.special_layers < 0 and i < -self.special_layers:
                kernel_sizes_conv.append((3, 1, 1))
            else:
                kernel_sizes_conv.append((3, 3, 3))

        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        ResBottleneckBlock(1, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'decoder',
                        ResBottleneckBlock(base_channels, base_channels, kernel_sizes_conv[0]))
            else:
                setattr(self, f'encode{i}',
                        ResBottleneckBlock(base_channels, base_channels, kernel_sizes_conv[i]))

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = getattr(self, f'encode0')(x)
        x_1 = x
        for i in range(1, self.depth):
            if self.special_layers > 0 and i < self.special_layers:
                x_1 = self.max_pool_flat(x_1)
            elif self.special_layers < 0 and i < -self.special_layers:
                x_1 = self.max_pool_shrink(x_1)
            else:
                x_1 = self.max_pool(x_1)
            x_1 = getattr(self, f"encode{i}")(x_1)
            x = merge(x, x_1)
        x = self.out(x)
        return x