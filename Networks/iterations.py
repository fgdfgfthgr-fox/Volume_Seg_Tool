import torch
import torch.nn as nn
from .Modules import General_Components as Gen_Modules


# UNet with a dropout layer of p=0.25 after encode 5.
class Iteration_1(nn.Module):
    def __init__(self, base_channels=64, bn=True):
        super(Iteration_1, self).__init__()
        self.encode1 = Gen_Modules.DoubleConv3D(1, base_channels, bn=bn)
        self.encode2 = Gen_Modules.Down3D(base_channels, 2 * base_channels, bn=bn)
        self.encode3 = Gen_Modules.Down3D(2 * base_channels, 4 * base_channels, bn=bn)
        self.encode4 = Gen_Modules.Down3D(4 * base_channels, 8 * base_channels, bn=bn)
        self.encode5 = Gen_Modules.Down3D(8 * base_channels, 16 * base_channels, bn=bn)
        self.dropout = nn.Dropout3d(0.25)

        self.decode5 = Gen_Modules.Up3D(16 * base_channels, 8 * base_channels, bn=bn)
        self.decode4 = Gen_Modules.Up3D(8 * base_channels, 4 * base_channels, bn=bn)
        self.decode3 = Gen_Modules.Up3D(4 * base_channels, 2 * base_channels, bn=bn)
        self.decode2 = Gen_Modules.Up3D(2 * base_channels, base_channels, bn=bn)
        self.decode1 = Gen_Modules.DoubleConv3D(base_channels, base_channels, bn=bn)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        x1 = self.encode1(input)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)
        x5 = self.dropout(x5)

        x = self.decode5(x5, x4)
        x = self.decode4(x, x3)
        x = self.decode3(x, x2)
        x = self.decode2(x, x1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# UNet with a dropout layer of p=0.5 after encode 5.
class Iteration_1_1(nn.Module):
    def __init__(self, base_channels=64, bn=True):
        super(Iteration_1_1, self).__init__()
        self.encode1 = Gen_Modules.DoubleConv3D(1, base_channels, bn=bn)
        self.encode2 = Gen_Modules.Down3D(base_channels, 2 * base_channels, bn=bn)
        self.encode3 = Gen_Modules.Down3D(2 * base_channels, 4 * base_channels, bn=bn)
        self.encode4 = Gen_Modules.Down3D(4 * base_channels, 8 * base_channels, bn=bn)
        self.encode5 = Gen_Modules.Down3D(8 * base_channels, 16 * base_channels, bn=bn)
        self.dropout = nn.Dropout3d(0.5)

        self.decode5 = Gen_Modules.Up3D(16 * base_channels, 8 * base_channels, bn=bn)
        self.decode4 = Gen_Modules.Up3D(8 * base_channels, 4 * base_channels, bn=bn)
        self.decode3 = Gen_Modules.Up3D(4 * base_channels, 2 * base_channels, bn=bn)
        self.decode2 = Gen_Modules.Up3D(2 * base_channels, base_channels, bn=bn)
        self.decode1 = Gen_Modules.DoubleConv3D(base_channels, base_channels, bn=bn)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        x1 = self.encode1(input)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)
        x5 = self.dropout(x5)

        x = self.decode5(x5, x4)
        x = self.decode4(x, x3)
        x = self.decode3(x, x2)
        x = self.decode2(x, x1)
        x = self.decode1(x)

        output = self.out(x)
        return output