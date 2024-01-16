import torch
import torch.nn as nn
import torch.nn.functional as F

class IniConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5, 5), bn=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2,
                   (kernel_size[1] - 1) // 2,
                   (kernel_size[2] - 1) // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.prelu = nn.PReLU()
        if in_channels != out_channels:
            self.identity = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.identity = None

    def forward(self, x):
        x_id = x
        x = self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        if self.identity:
            x_id = self.identity(x_id)
        x = x + x_id
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), bn=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.prelu = nn.PReLU()
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        return x


class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5, 5), bn=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2,
                   (kernel_size[1] - 1) // 2,
                   (kernel_size[2] - 1) // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.prelu = nn.PReLU()
        if in_channels != out_channels:
            self.identity = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.identity = None

    def forward(self, x):
        x_id = x
        x = self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        if self.identity:
            x_id = self.identity(x_id)
        x = x + x_id
        return x


class ThreeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5, 5), bn=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2,
                   (kernel_size[1] - 1) // 2,
                   (kernel_size[2] - 1) // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.prelu = nn.PReLU()
        if in_channels != out_channels:
            self.identity = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.identity = None

    def forward(self, x):
        x_id = x
        x = self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        x = self.conv3(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        if self.identity:
            x_id = self.identity(x_id)
        x = x + x_id
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), bn=True):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride)
        self.prelu = nn.PReLU()
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        return x


class IniConvUp(nn.Module):
    def __init__(self, up_in, in_channels, out_channels, kernel_size=(5, 5, 5), bn=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2,
                   (kernel_size[1] - 1) // 2,
                   (kernel_size[2] - 1) // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.prelu = nn.PReLU()
        if up_in != out_channels:
            self.identity = nn.Conv3d(up_in, out_channels, kernel_size=1)
        else:
            self.identity = None

    def forward(self, x, x1):
        x_id = x
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        if self.identity:
            x_id = self.identity(x_id)
        x = x + x_id
        return x


class TwoConvUp(nn.Module):
    def __init__(self, up_in, in_channels, out_channels, kernel_size=(5, 5, 5), bn=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2,
                   (kernel_size[1] - 1) // 2,
                   (kernel_size[2] - 1) // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.prelu = nn.PReLU()
        if up_in != out_channels:
            self.identity = nn.Conv3d(up_in, out_channels, kernel_size=1)
        else:
            self.identity = None

    def forward(self, x, x1):
        x_id = x
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        if self.identity:
            x_id = self.identity(x_id)
        x = x + x_id
        return x


class ThreeConvUp(nn.Module):
    def __init__(self, up_in, in_channels, out_channels, kernel_size=(5, 5, 5), bn=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2,
                   (kernel_size[1] - 1) // 2,
                   (kernel_size[2] - 1) // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.prelu = nn.PReLU()
        if up_in != out_channels:
            self.identity = nn.Conv3d(up_in, out_channels, kernel_size=1)
        else:
            self.identity = None

    def forward(self, x, x1):
        x_id = x
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        x = self.conv3(x)
        if self.bn:
            x = self.bn(x)
        x = self.prelu(x)
        if self.identity:
            x_id = self.identity(x_id)
        x = x + x_id
        return x