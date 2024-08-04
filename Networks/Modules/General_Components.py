import torch.nn as nn
import math


class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros', bias=False)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros')
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.celu = nn.CELU(inplace=True)

    def forward(self, x):
        x_id = x
        x = self.celu(self.bn(self.conv_1(x)))
        x = self.conv_2(x)
        x_id = self.mapping(x_id) if self.mapping else x_id
        x = self.celu(self.bn(x + x_id))
        return x


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), neck=4):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        middle_channel = math.ceil(out_channels/neck)
        self.conv_1 = nn.Conv3d(in_channels, middle_channel, kernel_size=1, bias=False)
        self.conv_2 = nn.Conv3d(middle_channel, middle_channel, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros', bias=False)
        self.conv_3 = nn.Conv3d(middle_channel, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(middle_channel, track_running_stats=False)
        self.bn2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.celu = nn.CELU(inplace=True)

    def forward(self, x):
        x_id = x
        x = self.celu(self.bn1(self.conv_1(x)))
        x = self.celu(self.bn1(self.conv_2(x)))
        x = self.conv_3(x)
        x_id = self.mapping(x_id) if self.mapping else x_id
        x = self.celu(self.bn2(x + x_id))
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv_1 = nn.Conv3d(in_channels, out_channels,
                                kernel_size=kernel_size, padding=padding, padding_mode='zeros', bias=False)
        self.conv_2 = nn.Conv3d(out_channels, out_channels,
                                kernel_size=kernel_size, padding=padding, padding_mode='zeros', bias=False)
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.celu = nn.CELU(inplace=True)

    def forward(self, x):
        x = self.celu(self.bn(self.conv_1(x)))
        x = self.celu(self.bn(self.conv_2(x)))
        return x


class BasicBlockSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv_1 = nn.Conv3d(in_channels, out_channels,
                                kernel_size=kernel_size, padding=padding, padding_mode='zeros', bias=False)
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.celu = nn.CELU(inplace=True)

    def forward(self, x):
        x = self.celu(self.bn(self.conv_1(x)))
        return x


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.celu = nn.CELU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 2, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        x_avg = self.avg_pool(x).view(batch_size, num_channels)
        x_avg = self.fc1(x_avg)
        x_avg = self.celu(x_avg)
        x_avg = self.fc2(x_avg)
        x_avg = self.sigmoid(x_avg)
        return x * (x_avg.view(batch_size, num_channels, 1, 1, 1))


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_spatial = self.fc(x)
        x_spatial = self.sigmoid(x_spatial)
        return x * x_spatial


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cse = cSE(in_channels)
        self.sse = sSE(in_channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)
