import torch.nn as nn
import math
from .Modules.General_Components import ResBasicBlock, ResBottleneckBlock, BasicBlock

# Models for various testing purpose


class Tiniest(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1):
        super(Tiniest, self).__init__()
        self.model = BasicBlock(1, base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        x = self.model(input)

        output = self.out(x)

        return output