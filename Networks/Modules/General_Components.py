import torch.nn as nn
import math


class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros')
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros')
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_id = x
        x = self.relu(self.bn(self.conv_1(x)))
        x = self.conv_2(x)
        x_id = self.mapping(x_id) if self.mapping else x_id
        x = x + x_id
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), neck=4):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        middle_channel = math.ceil(out_channels/neck)
        self.conv_1 = nn.Conv3d(in_channels, middle_channel, kernel_size=1)
        self.conv_2 = nn.Conv3d(middle_channel, middle_channel, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros')
        self.conv_3 = nn.Conv3d(middle_channel, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(middle_channel, track_running_stats=False)
        self.bn2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_id = x
        x = self.relu(self.bn1(self.conv_1(x)))
        x = self.relu(self.bn1(self.conv_2(x)))
        x = self.conv_3(x)
        x_id = self.mapping(x_id) if self.mapping else x_id
        x = x + x_id
        x = self.bn2(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv_1 = nn.Conv3d(in_channels, out_channels,
                                kernel_size=kernel_size, padding=padding, padding_mode='zeros')
        self.conv_2 = nn.Conv3d(out_channels, out_channels,
                                kernel_size=kernel_size, padding=padding, padding_mode='zeros')
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv_1(x)))
        x = self.relu(self.bn(self.conv_2(x)))
        return x