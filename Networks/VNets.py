import torch
import torch.nn as nn
from .Modules import VNets_Components as Modules

# F. Milletari, N. Navab, S.-A. Ahmadi, V-Net: fully convolutional neural networks for volumetric medical image
# segmentation, in: 2016 Fourth International Conference on 3D Vision (3DV), IEEE, 2016, pp. 565–571,
# doi:10.1109/3DV.2016.79.

# The original VNet implementation in the paper doesn't use batch normalization

class VNet(nn.Module):
    def __init__(self, base_channels=16, bn=True):
        super(VNet, self).__init__()
        self.encode1 = Modules.IniConv(1, base_channels, bn=bn)
        self.down1 = Modules.DownConv(base_channels, 2 * base_channels, bn=bn)
        self.encode2 = Modules.TwoConv(2 * base_channels, 2 * base_channels, bn=bn)
        self.down2 = Modules.DownConv(2 * base_channels, 4 * base_channels, bn=bn)
        self.encode3 = Modules.ThreeConv(4 * base_channels, 4 * base_channels, bn=bn)
        self.down3 = Modules.DownConv(4 * base_channels, 8 * base_channels, bn=bn)
        self.encode4 = Modules.ThreeConv(8 * base_channels, 8 * base_channels, bn=bn)
        self.down4 = Modules.DownConv(8 * base_channels, 16 * base_channels, bn=bn)
        self.encode5 = Modules.ThreeConv(16 * base_channels, 16 * base_channels, bn=bn)

        self.up4 = Modules.UpConv(16 * base_channels, 16 * base_channels, bn=bn)
        self.decode4 = Modules.ThreeConvUp(16 * base_channels, (16+8) * base_channels, 16 * base_channels, bn=bn)
        self.up3 = Modules.UpConv(16 * base_channels, 8 * base_channels, bn=bn)
        self.decode3 = Modules.ThreeConvUp(8 * base_channels, (8+4) * base_channels, 8 * base_channels, bn=bn)
        self.up2 = Modules.UpConv(8 * base_channels, 4 * base_channels, bn=bn)
        self.decode2 = Modules.TwoConvUp(4 * base_channels, (4+2) * base_channels, 4 * base_channels, bn=bn)
        self.up1 = Modules.UpConv(4 * base_channels, 2 * base_channels, bn=bn)
        self.decode1 = Modules.IniConvUp(2 * base_channels, (2+1) * base_channels, 2 * base_channels, bn=bn)

        self.out = nn.Sequential(nn.Conv3d(base_channels * 2, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        x1 = self.encode1(input)
        x2 = self.down1(x1)
        x2 = self.encode2(x2)
        x3 = self.down2(x2)
        x3 = self.encode3(x3)
        x4 = self.down3(x3)
        x4 = self.encode4(x4)
        x5 = self.down4(x4)
        x5 = self.encode5(x5)

        x = self.up4(x5)
        x = self.decode4(x, x4)
        x = self.up3(x)
        x = self.decode3(x, x3)
        x = self.up2(x)
        x = self.decode2(x, x2)
        x = self.up1(x)
        x = self.decode1(x, x1)

        output = self.out(x)
        return output


# VNet with the depth of three, in case the full size one is unnecessary
class VNet_Lite(nn.Module):
    def __init__(self, base_channels=16, bn=True):
        super(VNet_Lite, self).__init__()
        self.encode1 = Modules.IniConv(1, base_channels, bn=bn)
        self.down1 = Modules.DownConv(base_channels, 2 * base_channels, bn=bn)
        self.encode2 = Modules.TwoConv(2 * base_channels, 2 * base_channels, bn=bn)
        self.down2 = Modules.DownConv(2 * base_channels, 4 * base_channels, bn=bn)

        self.bottleneck = Modules.ThreeConv(4 * base_channels, 4 * base_channels, bn=bn)

        self.up2 = Modules.UpConv(4 * base_channels, 4 * base_channels, bn=bn)
        self.decode2 = Modules.TwoConvUp(4 * base_channels, (4+2) * base_channels, 4 * base_channels, bn=bn)
        self.up1 = Modules.UpConv(4 * base_channels, 2 * base_channels, bn=bn)
        self.decode1 = Modules.IniConvUp(2 * base_channels, (2+1) * base_channels, 2 * base_channels, bn=bn)

        self.out = nn.Sequential(nn.Conv3d(base_channels * 2, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        x1 = self.encode1(input)
        x2 = self.down1(x1)
        x2 = self.encode2(x2)
        x3 = self.down2(x2)

        x = self.bottleneck(x3)

        x = self.up2(x)
        x = self.decode2(x, x2)
        x = self.up1(x)
        x = self.decode1(x, x1)

        output = self.out(x)
        return output