import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1), bn=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, padding_mode='replicate',
                               dilation=dilation, bias=False)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, stride=stride,
                               kernel_size=kernel_size, padding=padding, padding_mode='replicate',
                               dilation=dilation, bias=False)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), method="max-pool", dilation=(1, 1, 1)):
        super().__init__()
        if method == "max-pool":
            self.steps = nn.Sequential(
                nn.MaxPool3d(
                    kernel_size=pool_kernel_size,
                    ceil_mode=True,
                ),
                DoubleConv3D(in_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn)
            )
        elif method == "avg-pool":
            self.steps = nn.Sequential(
                nn.AvgPool3d(
                    kernel_size=pool_kernel_size,
                    ceil_mode=True,
                ),
                DoubleConv3D(in_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn)
            )

    def forward(self, x):
        return self.steps(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), scale_kernel_size=(2, 2, 2), method="transpose"):
        super().__init__()
        if method == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=in_channels // 2,
                kernel_size=scale_kernel_size, stride=scale_kernel_size
            )
        self.method = method
        self.scale_kernel_size = scale_kernel_size
        self.conv = DoubleConv3D(in_channels, out_channels, conv_kernel_size, bn=bn)

    def forward(self, x1, x2):
        if self.method == "transpose":
            x1 = self.up(x1)
        elif self.method == "interpolation":
            x1 = F.interpolate(x1, scale_factor=self.scale_kernel_size, mode="trilinear", align_corners=True)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResDoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1), bn=True):
        super(ResDoubleConv3D, self).__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode='replicate', dilation=dilation, bias=False)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='replicate', dilation=dilation, bias=False)
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


class ResBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), dilation=(1, 1, 1), neck=4, cardinality=1, bn=False):
        super(ResBottleNeck, self).__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn_1 = nn.BatchNorm3d(out_channels//neck, track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.conv_1 = nn.Conv3d(in_channels, out_channels//neck, kernel_size=1, stride=1, bias=False)
        self.conv_2 = nn.Conv3d(out_channels//neck, out_channels//neck, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=cardinality, padding_mode='replicate', bias=False)
        self.conv_3 = nn.Conv3d(out_channels//neck, out_channels, kernel_size=1, stride=1, bias=False)
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


class ResDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bottleneck=False,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        if bottleneck:
            self.steps = nn.Sequential(
                nn.MaxPool3d(
                    kernel_size=pool_kernel_size,
                    ceil_mode=True,
                ),
                ResBottleNeck(in_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                              neck=neck, cardinality=cardinality)
            )
        else:
            self.steps = nn.Sequential(
                nn.MaxPool3d(
                    kernel_size=pool_kernel_size,
                    ceil_mode=True,
                ),
                ResDoubleConv3D(in_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn)
            )

    def forward(self, x):
        return self.steps(x)


class ResUp3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bottleneck=False, dilation=(1,1,1),
                 conv_kernel_size=(3, 3, 3), scale_kernel_size=(2, 2, 2), method="transpose", neck=4, cardinality=1):
        super().__init__()
        if method == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=scale_kernel_size, stride=scale_kernel_size
            )
        self.method = method
        self.scale_kernel_size = scale_kernel_size
        if bottleneck:
            self.conv = ResBottleNeck(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation,
                                      neck=neck, cardinality=cardinality)
        else:
            self.conv = ResDoubleConv3D(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation)

    def forward(self, x1, x2):
        if self.method == "transpose":
            x1 = self.up(x1)
        elif self.method == "interpolation":
            x1 = F.interpolate(x1, scale_factor=self.scale_kernel_size, mode="trilinear", align_corners=True)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MaxBlurPool3D(nn.Module):
    def __init__(self, kernel_size=2, blur_kernel_size=3, sigma=1, gauss_kernel=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.max = nn.MaxPool3d(kernel_size=kernel_size, stride=1)
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma
        if gauss_kernel:
            self.blur_kernel = self.create_blur_kernel_gaussian()
        else:
            self.blur_kernel = self.create_blur_kernel()

    def create_blur_kernel_gaussian(self):
        x, y, z = np.meshgrid(
            np.arange(-self.blur_kernel_size // 2 + 1, self.blur_kernel_size // 2 + 1),
            np.arange(-self.blur_kernel_size // 2 + 1, self.blur_kernel_size // 2 + 1),
            np.arange(-self.blur_kernel_size // 2 + 1, self.blur_kernel_size // 2 + 1)
        )
        kernel = np.exp(
            -(x ** 2 / (2 * self.sigma ** 2) + y ** 2 / (2 * self.sigma ** 2) + z ** 2 / (2 * self.sigma ** 2)))
        kernel = kernel / kernel.sum()
        return torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)

    def create_blur_kernel(self):
        array = np.array((((1, 1, 1),
                           (1, 2, 1),
                           (1, 1, 1)),
                          ((1, 2, 1),
                           (2, 4, 2),
                           (1, 2, 1)),
                          ((1, 1, 1),
                           (1, 2, 1),
                           (1, 1, 1)),
                          ))
        array = array / array.sum()
        return torch.FloatTensor(array).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        x = F.pad(x,(0, 1, 0, 1, 0, 1), value=-math.inf)
        x = self.max(x)
        channels = x.shape[1]
        kernel = self.blur_kernel.to(x.device)
        kernel = kernel.expand(channels, channels, -1, -1, -1)
        x = F.conv3d(x, kernel, stride=self.kernel_size, padding=1)
        return x


class ResBlurDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=2, dilation=(1, 1, 1), gauss_kernel=True):
        super().__init__()
        self.steps = nn.Sequential(
            MaxBlurPool3D(
                kernel_size=pool_kernel_size,
                gauss_kernel=gauss_kernel
            ),
            ResBottleNeck(in_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn)
        )

    def forward(self, x):
        return self.steps(x)


class cSE3D(nn.Module):
    def __init__(self, in_channels):
        super(cSE3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_avg = self.fc1(x_avg)
        x_avg = self.relu(x_avg)
        x_avg = self.fc2(x_avg)
        x_avg = self.sigmoid(x_avg)
        return x * x_avg

class sSE3D(nn.Module):
    def __init__(self, in_channels):
        super(sSE3D, self).__init__()
        self.fc = nn.Conv3d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_spatial = self.fc(x)
        x_spatial = self.sigmoid(x_spatial)
        return x * x_spatial

class scSE3D(nn.Module):
    def __init__(self, in_channels):
        super(scSE3D, self).__init__()
        self.cse = cSE3D(in_channels)
        self.sse = sSE3D(in_channels)

    def forward(self, x):
        cse_output = self.cse(x)
        sse_output = self.sse(x)
        return cse_output + sse_output


class ResBottleNeckAsy(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=(1, 1, 1), bn=False):
        super(ResBottleNeckAsy, self).__init__()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn_1 = nn.BatchNorm3d(out_channels//4, track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.conv_1 = nn.Conv3d(in_channels, out_channels//4, kernel_size=1, stride=1, bias=False)
        self.conv_2_1 = nn.Conv3d(out_channels//4, out_channels//4, kernel_size=(3,1,1), stride=stride,
                                  padding=(1,0,0), padding_mode='replicate', bias=False)
        self.conv_2_2 = nn.Conv3d(out_channels//4, out_channels//4, kernel_size=(1,3,1), stride=stride,
                                  padding=(0,1,0), padding_mode='replicate', bias=False)
        self.conv_2_3 = nn.Conv3d(out_channels//4, out_channels//4, kernel_size=(1,1,3), stride=stride,
                                  padding=(0,0,1), padding_mode='replicate', bias=False)
        self.conv_3 = nn.Conv3d(out_channels//4, out_channels, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_id = x
        x = self.conv_1(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.conv_2_3(x)
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


class ResDownAsy3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1)):
        super().__init__()
        self.steps = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                ceil_mode=True,
            ),
            ResBottleNeckAsy(in_channels, out_channels, bn=bn,)
        )

    def forward(self, x):
        return self.steps(x)


class ResUpAsy3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, scale_kernel_size=(2, 2, 2)):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=in_channels // 2,
            kernel_size=scale_kernel_size, stride=scale_kernel_size
        )
        self.scale_kernel_size = scale_kernel_size
        self.conv = ResBottleNeckAsy(in_channels, out_channels, bn=bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=16, drop_rate=0.0, bn=True):
        super(DenseLayer, self).__init__()
        if bn:
            self.bn = nn.BatchNorm3d(in_channels, track_running_stats=False)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, growth_rate,
                              kernel_size=3, stride=1, padding=1,
                              bias=False)
        if drop_rate != 0.0:
            self.dropout = nn.Dropout3d(drop_rate)
        else:
            self.dropout = None

    def forward(self, x):
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16, drop_rate=0.0, num_layers=4, bn=True, upsample=False):
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


class DenseDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, num_layers=4, pool_kernel_size=(2, 2, 2),):
        super().__init__()
        self.steps = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                ceil_mode=True,
            ),
            DenseBlock(in_channels, (out_channels-in_channels)//num_layers, bn=bn, num_layers=num_layers)
        )

    def forward(self, x):
        return self.steps(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = input.contiguous().view(batch_size, nOut, self.upscale_factor, self.upscale_factor, self.upscale_factor, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class ResUpSubpixel3D(nn.Module):
    def __init__(self, in_channels, bn=True, conv_kernel_size=(3, 3, 3), pre_shuffle_mult=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels*pre_shuffle_mult, kernel_size=1, bias=False)
        self.shuffle = PixelShuffle3d(2)
        self.conv2 = ResBottleNeck(in_channels//4*pre_shuffle_mult, in_channels//8*pre_shuffle_mult, conv_kernel_size, bn=bn)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.shuffle(x1)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv2(x)


class Merge3D(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        super(Merge3D, self).__init__()
        self.scaleup = torch.nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True)
        if in_channels != out_channels:
            self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.conv = None

    def forward(self, x1, x2):
        if self.conv:
            x1 = self.conv(x1)
        x1 = self.scaleup(x1)
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


class ResConvDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1)):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=pool_kernel_size, stride=2),
            ResBottleNeck(in_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn)
        )

    def forward(self, x):
        return self.steps(x)


class ResConvDownAlt3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1)):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=pool_kernel_size, stride=2),
            nn.ReLU(),
            ResBottleNeck(in_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn)
        )

    def forward(self, x):
        return self.steps(x)


class ResDown3D_It13_0(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                ceil_mode=True,
            ),
            ResBottleNeck(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                              neck=neck, cardinality=cardinality)
        )

    def forward(self, x):
        return self.steps(x)


class ResDown3D_It13_2(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size, ceil_mode=True)
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=pool_kernel_size, stride=2)
        self.steps = ResBottleNeck(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                                   neck=neck, cardinality=cardinality)

    def forward(self, x):
        x = torch.cat((self.pool(x), self.conv(x)), dim=1)
        x = self.steps(x)
        return x


class ResDown3D_It13_3(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size, ceil_mode=True)
        self.pad = nn.ReplicationPad3d((0, 1, 0, 1, 0, 1))
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2)
        self.steps = ResBottleNeck(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                                   neck=neck, cardinality=cardinality)

    def forward(self, x):
        pool_result = self.pool(x)
        conv_result = self.conv(self.pad(x))
        x = torch.cat((pool_result, conv_result), dim=1)
        x = self.steps(x)
        return x


class ResDown3D_It13_4(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size, ceil_mode=True)
        self.pad = nn.ReplicationPad3d((0, 1, 0, 1, 0, 1))
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.steps = ResBottleNeck(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                                   neck=neck, cardinality=cardinality)

    def forward(self, x):
        pool_result = self.pool(x)
        conv_result = self.relu(self.conv(self.pad(x)))
        x = torch.cat((pool_result, conv_result), dim=1)
        x = self.steps(x)
        return x


class ResBottleNeckCeil(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), dilation=(1, 1, 1), neck=4, cardinality=1, bn=False):
        super(ResBottleNeckCeil, self).__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn_1 = nn.BatchNorm3d(math.ceil(out_channels/neck), track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.conv_1 = nn.Conv3d(in_channels, math.ceil(out_channels/neck), kernel_size=1, stride=1, bias=False)
        self.conv_2 = nn.Conv3d(math.ceil(out_channels/neck), math.ceil(out_channels/neck), kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=cardinality, padding_mode='replicate', bias=False)
        self.conv_3 = nn.Conv3d(math.ceil(out_channels/neck), out_channels, kernel_size=1, stride=1, bias=False)
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


class ResDown3D_It13_6(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                ceil_mode=True,
            ),
            ResBottleNeckCeil(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                              neck=neck, cardinality=cardinality)
        )

    def forward(self, x):
        return self.steps(x)


class ResUp3DCeil(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bottleneck=False, dilation=(1,1,1),
                 conv_kernel_size=(3, 3, 3), scale_kernel_size=(2, 2, 2), method="transpose", neck=4, cardinality=1):
        super().__init__()
        if method == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=scale_kernel_size, stride=scale_kernel_size
            )
        self.method = method
        self.scale_kernel_size = scale_kernel_size
        if bottleneck:
            self.conv = ResBottleNeckCeil(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation,
                                      neck=neck, cardinality=cardinality)
        else:
            self.conv = ResDoubleConv3D(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation)

    def forward(self, x1, x2):
        if self.method == "transpose":
            x1 = self.up(x1)
        elif self.method == "interpolation":
            x1 = F.interpolate(x1, scale_factor=self.scale_kernel_size, mode="trilinear", align_corners=True)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResBottleNeckCeilELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), dilation=(1, 1, 1), neck=4, cardinality=1, bn=False):
        super(ResBottleNeckCeilELU, self).__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn_1 = nn.BatchNorm3d(math.ceil(out_channels/neck), track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.conv_1 = nn.Conv3d(in_channels, math.ceil(out_channels/neck), kernel_size=1, stride=1, bias=False)
        self.conv_2 = nn.Conv3d(math.ceil(out_channels/neck), math.ceil(out_channels/neck), kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=cardinality, padding_mode='replicate', bias=False)
        self.conv_3 = nn.Conv3d(math.ceil(out_channels/neck), out_channels, kernel_size=1, stride=1, bias=False)
        self.elu = nn.ELU()

    def forward(self, x):
        x_id = x
        x = self.conv_1(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.elu(x)
        x = self.conv_2(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.elu(x)
        x = self.conv_3(x)
        if self.bn_2:
            x = self.bn_2(x)
        x = self.elu(x)
        if self.downsample:
            x_id = self.downsample(x_id)
        if self.bn_2:
            x_id = self.bn_2(x_id)
        x = x + x_id
        if self.bn_2:
            x = self.bn_2(x)
        x = self.elu(x)
        return x


class ResDown3D_It15_0(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                ceil_mode=True,
            ),
            ResBottleNeckCeilELU(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                              neck=neck, cardinality=cardinality)
        )

    def forward(self, x):
        return self.steps(x)


class ResUp3DCeilELU(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bottleneck=False, dilation=(1,1,1),
                 conv_kernel_size=(3, 3, 3), scale_kernel_size=(2, 2, 2), method="transpose", neck=4, cardinality=1):
        super().__init__()
        if method == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=scale_kernel_size, stride=scale_kernel_size
            )
        self.method = method
        self.scale_kernel_size = scale_kernel_size
        if bottleneck:
            self.conv = ResBottleNeckCeilELU(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation,
                                      neck=neck, cardinality=cardinality)
        else:
            self.conv = ResDoubleConv3D(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation)

    def forward(self, x1, x2):
        if self.method == "transpose":
            x1 = self.up(x1)
        elif self.method == "interpolation":
            x1 = F.interpolate(x1, scale_factor=self.scale_kernel_size, mode="trilinear", align_corners=True)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResBottleNeckCeilPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), dilation=(1, 1, 1), neck=4, cardinality=1, bn=False):
        super(ResBottleNeckCeilPReLU, self).__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn_1 = nn.BatchNorm3d(math.ceil(out_channels/neck), track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.conv_1 = nn.Conv3d(in_channels, math.ceil(out_channels/neck), kernel_size=1, stride=1, bias=False)
        self.conv_2 = nn.Conv3d(math.ceil(out_channels/neck), math.ceil(out_channels/neck), kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=cardinality, padding_mode='replicate', bias=False)
        self.conv_3 = nn.Conv3d(math.ceil(out_channels/neck), out_channels, kernel_size=1, stride=1, bias=False)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x_id = x
        x = self.conv_1(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.prelu(x)
        x = self.conv_2(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.prelu(x)
        x = self.conv_3(x)
        if self.bn_2:
            x = self.bn_2(x)
        x = self.prelu(x)
        if self.downsample:
            x_id = self.downsample(x_id)
        if self.bn_2:
            x_id = self.bn_2(x_id)
        x = x + x_id
        if self.bn_2:
            x = self.bn_2(x)
        x = self.prelu(x)
        return x


class ResDown3D_It15_1(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                ceil_mode=True,
            ),
            ResBottleNeckCeilPReLU(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                                   neck=neck, cardinality=cardinality)
        )

    def forward(self, x):
        return self.steps(x)


class ResUp3DCeilPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bottleneck=False, dilation=(1,1,1),
                 conv_kernel_size=(3, 3, 3), scale_kernel_size=(2, 2, 2), method="transpose", neck=4, cardinality=1):
        super().__init__()
        if method == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=scale_kernel_size, stride=scale_kernel_size
            )
        self.method = method
        self.scale_kernel_size = scale_kernel_size
        if bottleneck:
            self.conv = ResBottleNeckCeilPReLU(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation,
                                      neck=neck, cardinality=cardinality)
        else:
            self.conv = ResDoubleConv3D(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation)

    def forward(self, x1, x2):
        if self.method == "transpose":
            x1 = self.up(x1)
        elif self.method == "interpolation":
            x1 = F.interpolate(x1, scale_factor=self.scale_kernel_size, mode="trilinear", align_corners=True)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResBottleNeckCeilRReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), dilation=(1, 1, 1), neck=4, cardinality=1, bn=False):
        super(ResBottleNeckCeilRReLU, self).__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn_1 = nn.BatchNorm3d(math.ceil(out_channels/neck), track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.conv_1 = nn.Conv3d(in_channels, math.ceil(out_channels/neck), kernel_size=1, stride=1, bias=False)
        self.conv_2 = nn.Conv3d(math.ceil(out_channels/neck), math.ceil(out_channels/neck), kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=cardinality, padding_mode='replicate', bias=False)
        self.conv_3 = nn.Conv3d(math.ceil(out_channels/neck), out_channels, kernel_size=1, stride=1, bias=False)
        self.rrelu = nn.RReLU()

    def forward(self, x):
        x_id = x
        x = self.conv_1(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.rrelu(x)
        x = self.conv_2(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.rrelu(x)
        x = self.conv_3(x)
        if self.bn_2:
            x = self.bn_2(x)
        x = self.rrelu(x)
        if self.downsample:
            x_id = self.downsample(x_id)
        if self.bn_2:
            x_id = self.bn_2(x_id)
        x = x + x_id
        if self.bn_2:
            x = self.bn_2(x)
        x = self.rrelu(x)
        return x


class ResDown3D_It15_2(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                ceil_mode=True,
            ),
            ResBottleNeckCeilRReLU(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                                   neck=neck, cardinality=cardinality)
        )

    def forward(self, x):
        return self.steps(x)


class ResUp3DCeilRReLU(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bottleneck=False, dilation=(1,1,1),
                 conv_kernel_size=(3, 3, 3), scale_kernel_size=(2, 2, 2), method="transpose", neck=4, cardinality=1):
        super().__init__()
        if method == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=scale_kernel_size, stride=scale_kernel_size
            )
        self.method = method
        self.scale_kernel_size = scale_kernel_size
        if bottleneck:
            self.conv = ResBottleNeckCeilRReLU(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation,
                                               neck=neck, cardinality=cardinality)
        else:
            self.conv = ResDoubleConv3D(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation)

    def forward(self, x1, x2):
        if self.method == "transpose":
            x1 = self.up(x1)
        elif self.method == "interpolation":
            x1 = F.interpolate(x1, scale_factor=self.scale_kernel_size, mode="trilinear", align_corners=True)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResBottleNeckCeilGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), dilation=(1, 1, 1), neck=4, cardinality=1, bn=False):
        super(ResBottleNeckCeilGELU, self).__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn_1 = nn.BatchNorm3d(math.ceil(out_channels/neck), track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.conv_1 = nn.Conv3d(in_channels, math.ceil(out_channels/neck), kernel_size=1, stride=1, bias=False)
        self.conv_2 = nn.Conv3d(math.ceil(out_channels/neck), math.ceil(out_channels/neck), kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=cardinality, padding_mode='replicate', bias=False)
        self.conv_3 = nn.Conv3d(math.ceil(out_channels/neck), out_channels, kernel_size=1, stride=1, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        x_id = x
        x = self.conv_1(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.gelu(x)
        x = self.conv_2(x)
        if self.bn_1:
            x = self.bn_1(x)
        x = self.gelu(x)
        x = self.conv_3(x)
        if self.bn_2:
            x = self.bn_2(x)
        x = self.gelu(x)
        if self.downsample:
            x_id = self.downsample(x_id)
        if self.bn_2:
            x_id = self.bn_2(x_id)
        x = x + x_id
        if self.bn_2:
            x = self.bn_2(x)
        x = self.gelu(x)
        return x


class ResDown3D_It15_3(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                ceil_mode=True,
            ),
            ResBottleNeckCeilGELU(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                                   neck=neck, cardinality=cardinality)
        )

    def forward(self, x):
        return self.steps(x)


class ResUp3DCeilGELU(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bottleneck=False, dilation=(1,1,1),
                 conv_kernel_size=(3, 3, 3), scale_kernel_size=(2, 2, 2), method="transpose", neck=4, cardinality=1):
        super().__init__()
        if method == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=scale_kernel_size, stride=scale_kernel_size
            )
        self.method = method
        self.scale_kernel_size = scale_kernel_size
        if bottleneck:
            self.conv = ResBottleNeckCeilGELU(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation,
                                              neck=neck, cardinality=cardinality)
        else:
            self.conv = ResDoubleConv3D(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation)

    def forward(self, x1, x2):
        if self.method == "transpose":
            x1 = self.up(x1)
        elif self.method == "interpolation":
            x1 = F.interpolate(x1, scale_factor=self.scale_kernel_size, mode="trilinear", align_corners=True)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResBottleNeckCeilDouble(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), dilation=(1, 1, 1), neck=4, cardinality=1, bn=False):
        super(ResBottleNeckCeilDouble, self).__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        if bn:
            self.bn_1 = nn.BatchNorm3d(math.ceil(out_channels/neck), track_running_stats=False)
            self.bn_2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn_1 = None
            self.bn_2 = None
        self.conv_1 = nn.Conv3d(in_channels, math.ceil(out_channels/neck), kernel_size=1, stride=1, bias=False)
        self.conv_2 = nn.Conv3d(math.ceil(out_channels/neck), math.ceil(out_channels/neck), kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=cardinality, padding_mode='replicate', bias=False)
        self.conv_2_1 = nn.Conv3d(math.ceil(out_channels/neck), math.ceil(out_channels/neck), kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=cardinality, padding_mode='replicate', bias=False)
        self.conv_3 = nn.Conv3d(math.ceil(out_channels/neck), out_channels, kernel_size=1, stride=1, bias=False)
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
        x = self.conv_2_1(x)
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



class ResDown3DDouble(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), dilation=(1, 1, 1), neck=4, cardinality=1):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                ceil_mode=True,
            ),
            ResBottleNeckCeilDouble(out_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn,
                                    neck=neck, cardinality=cardinality)
        )

    def forward(self, x):
        return self.steps(x)


class ResUp3DCeilDouble(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bottleneck=False, dilation=(1,1,1),
                 conv_kernel_size=(3, 3, 3), scale_kernel_size=(2, 2, 2), method="transpose", neck=4, cardinality=1):
        super().__init__()
        if method == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=scale_kernel_size, stride=scale_kernel_size
            )
        self.method = method
        self.scale_kernel_size = scale_kernel_size
        if bottleneck:
            self.conv = ResBottleNeckCeilDouble(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation,
                                                neck=neck, cardinality=cardinality)
        else:
            self.conv = ResDoubleConv3D(out_channels * 2, out_channels, conv_kernel_size, bn=bn, dilation=dilation)

    def forward(self, x1, x2):
        if self.method == "transpose":
            x1 = self.up(x1)
        elif self.method == "interpolation":
            x1 = F.interpolate(x1, scale_factor=self.scale_kernel_size, mode="trilinear", align_corners=True)
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)