import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), bn=False):
        super(ResBlock, self).__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode='replicate')
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='replicate')
        self.relu = nn.ReLU()

    def forward(self, x):
        x_id = x
        x = self.conv_1(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        if self.downsample:
            x_id = self.downsample(x_id)
        if self.bn:
            x_id = self.bn(x_id)
        x = self.conv_2(x)
        x = x + x_id
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ResBottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), bn=False):
        super(ResBottleNeckBlock, self).__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.downsample = None
        if bn:
            self.bn_1 = nn.BatchNorm3d(out_channels//4, track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.conv_1 = nn.Conv3d(in_channels, out_channels//4, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv3d(out_channels//4, out_channels//4, kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode='replicate')
        self.conv_3 = nn.Conv3d(out_channels//4, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_id = x
        x = self.conv_1(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_3(x)
        if self.bn_2:
            x = self.bn_2(x)
        x = self.relu(x)
        if self.downsample:
            x_id = self.downsample(x_id)
        if self.bn_2:
            x_id = self.bn_2(x_id)
        x = x + x_id
        if self.bn_2:
            x = self.bn_2(x)
        x = self.relu(x)
        return x