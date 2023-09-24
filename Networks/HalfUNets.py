import torch
import torch.nn as nn
from .Modules import HalfUNets_Components as Modules

# Lu, Haoran, et al. "Half-UNet: A simplified U-Net architecture for medical image segmentation." Frontiers in Neuroinformatics 16 (2022): 911679.
# Similar to UNet, but with a simplified decoder structure.
class HalfUNet(nn.Module):

    def __init__(self, base_channels=16, depth=5):
        super(HalfUNet, self).__init__()
        self.depth = depth
        self.inc = Modules.GhostDoubleConv3D(1, base_channels)

        self.down = nn.MaxPool3d(2)
        for i in range(1, depth+1):
            setattr(self, f'conv{i}', Modules.GhostDoubleConv3D(base_channels, base_channels))
            setattr(self, f'merge{i}', Modules.Merge3D((2 ** i, 2 ** i, 2 ** i)))

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = self.inc(x)
        x_1 = getattr(self, f'conv1')(x)
        x_1 = self.down(x_1)
        x = getattr(self, f'merge1')(x, x_1)
        if self.depth >= 2:
            for i in range(2, self.depth+1):
                x_1 = getattr(self, f'conv{i}')(x_1)
                x_1 = self.down(x_1)
                x = getattr(self, f'merge{i}')(x, x_1)
        x = self.out(x)
        return x