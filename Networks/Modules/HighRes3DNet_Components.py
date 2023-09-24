import torch
import torch.nn as nn
import torch.nn.functional as F


class HighRes3DNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), dilation=(1, 1, 1), bn=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding,
                               dilation=dilation)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding,
                               dilation=dilation)
        if in_channels != out_channels:
            self.identity = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.identity = None

        if bn:
            self.bn_1 = nn.BatchNorm3d(in_channels, track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.relu = nn.ReLU()

    def forward(self, x):
        x_id = x
        if self.bn_1:
            x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv1(x)
        if self.bn_2:
            x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.identity:
            x_id = self.identity(x_id)
        x = x + x_id
        return x