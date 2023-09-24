import torch
import torch.nn as nn
import torch.nn.functional as F


# https://blog.paperspace.com/ghostnet-cvpr-2020/
# Replacement for standard convolution, less parameter and faster in theory.
class GhostModule3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super(GhostModule3D, self).__init__()
        padding = ((kernel_size[0] - 1) // 2,
                   (kernel_size[1] - 1) // 2,
                   (kernel_size[2] - 1) // 2)
        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv3d(out_channels // 2, out_channels // 2, kernel_size=kernel_size, padding=padding,
                      groups=out_channels // 2),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat((x1, x2), dim=1)


class GhostDoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super(GhostDoubleConv3D, self).__init__()
        self.conv1 = GhostModule3D(in_channels, out_channels, kernel_size)
        self.conv2 = GhostModule3D(out_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Merge3D(nn.Module):
    def __init__(self, scale):
        super(Merge3D, self).__init__()
        self.scaleup = torch.nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True)

    def forward(self, x1, x2):
        x2 = self.scaleup(x2)
        # diffD = x2.size()[-3] - x1.size()[-3]
        # diffH = x2.size()[-2] - x1.size()[-2]
        # diffW = x2.size()[-1] - x1.size()[-1]
        # if not diffD == 0:
        #    x2 = x2[:, :, :-diffD, :, :]
        # if not diffH == 0:
        #    x2 = x2[:, :, :, :-diffH, :]
        # if not diffW == 0:
        #    x2 = x2[:, :, :, :, :-diffW]
        return torch.add(x1, x2)
