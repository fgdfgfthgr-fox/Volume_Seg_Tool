import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=16, drop_rate=0.2, bn=True):
        super(DenseLayer, self).__init__()
        if bn:
            self.bn = nn.BatchNorm3d(in_channels, track_running_stats=False)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, growth_rate,
                              kernel_size=3, stride=1, padding=1,
                              bias=True)
        if drop_rate != 0.0:
            self.dropout = nn.Dropout3d(drop_rate)

    def forward(self, x):
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16, drop_rate=0.2, num_layers=4, bn=True, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [DenseLayer(in_channels + i * growth_rate, growth_rate, drop_rate, bn)
                        for i in range(num_layers)]
        )

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x

class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, drop_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


class TransDown(nn.Module):
    def __init__(self, in_channels, drop_rate=0.2, bn=True):
        super(TransDown, self).__init__()
        if bn:
            self.bn = nn.BatchNorm3d(in_channels, track_running_stats=False)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        if drop_rate != 0.0:
            self.dropout = nn.Dropout3d(drop_rate)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        return x


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3), skip.size(4))
        out = torch.cat([out, skip], 1)
        return out


def center_crop(layer, max_depth, max_height, max_width):
    _, _, d, h, w = layer.size()
    xyz1 = (d - max_depth) // 2
    xyz2 = (h - max_height) // 2
    xyz3 = (w - max_width) // 2
    return layer[:, :, xyz1:(xyz1 + max_depth), xyz2:(xyz2 + max_height), xyz3:(xyz3 + max_width)]