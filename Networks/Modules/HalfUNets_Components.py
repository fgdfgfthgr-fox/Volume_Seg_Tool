import torch
import torch.nn as nn


# https://blog.paperspace.com/ghostnet-cvpr-2020/
# Replacement for standard convolution, less parameter and faster in theory.
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super(GhostModule, self).__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=1),
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


class GhostDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super(GhostDoubleConv, self).__init__()
        self.conv1 = GhostModule(in_channels, out_channels, kernel_size)
        self.conv2 = GhostModule(out_channels, out_channels, kernel_size)
#        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def merge(x1, x2):
    size = (x1.shape[2], x1.shape[3], x1.shape[4])
    x2 = torch.nn.functional.interpolate(x2, size=size, mode='trilinear', align_corners=True)
    return torch.add(x1, x2)
