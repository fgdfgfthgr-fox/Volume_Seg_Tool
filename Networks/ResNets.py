import torch
import torch.nn as nn
from .Modules import ResNets_Components as ResModules
from .Modules import General_Components as Modules


# He K. et al. (2016) Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778
# Introduced residual connections which greatly increased the depth of network without facing vanishing gradient issues.

# What's in this file are actually ResNets with a UNet-like decoder, as the original s doesn't have a decoder for image segmentation tasks.

class ResNet50(nn.Module):
    def __init__(self, base_channels=64, bn=True):
        super(ResNet50, self).__init__()
        # ResNet as encoder
        self.conv1 = nn.Conv3d(1, base_channels, kernel_size=7, stride=2, padding=3, padding_mode='replicate',
                               bias=False)
        if bn:
            self.bn = nn.BatchNorm3d(base_channels, track_running_stats=False)
        else:
            self.bn = None
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv2_1 = ResModules.ResBottleNeckBlock(base_channels, base_channels * 4, bn=bn)
        self.conv2_2 = ResModules.ResBottleNeckBlock(base_channels * 4, base_channels * 4, bn=bn)
        self.conv2_3 = ResModules.ResBottleNeckBlock(base_channels * 4, base_channels * 2, bn=bn)

        self.conv3_1 = ResModules.ResBottleNeckBlock(base_channels * 2, base_channels * 8, bn=bn, stride=2)
        self.conv3_2 = ResModules.ResBottleNeckBlock(base_channels * 8, base_channels * 8, bn=bn)
        self.conv3_3 = ResModules.ResBottleNeckBlock(base_channels * 8, base_channels * 8, bn=bn)
        self.conv3_4 = ResModules.ResBottleNeckBlock(base_channels * 8, base_channels * 4, bn=bn)

        self.conv4_1 = ResModules.ResBottleNeckBlock(base_channels * 4, base_channels * 16, bn=bn, stride=2)
        self.conv4_2 = ResModules.ResBottleNeckBlock(base_channels * 16, base_channels * 16, bn=bn)
        self.conv4_3 = ResModules.ResBottleNeckBlock(base_channels * 16, base_channels * 16, bn=bn)
        self.conv4_4 = ResModules.ResBottleNeckBlock(base_channels * 16, base_channels * 16, bn=bn)
        self.conv4_5 = ResModules.ResBottleNeckBlock(base_channels * 16, base_channels * 16, bn=bn)
        self.conv4_6 = ResModules.ResBottleNeckBlock(base_channels * 16, base_channels * 8, bn=bn)

        self.conv5_1 = ResModules.ResBottleNeckBlock(base_channels * 8, base_channels * 32, bn=bn, stride=2)
        self.conv5_2 = ResModules.ResBottleNeckBlock(base_channels * 32, base_channels * 32, bn=bn)
        self.conv5_3 = ResModules.ResBottleNeckBlock(base_channels * 32, base_channels * 16, bn=bn)
        # UNet-like decoder
        self.up_1 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8,
                                       kernel_size=2, stride=2)
        self.up_1_conv = Modules.DoubleConv3D(base_channels * 16, base_channels * 8)
        self.up_2 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4,
                                       kernel_size=2, stride=2)
        self.up_2_conv = Modules.DoubleConv3D(base_channels * 8, base_channels * 4)
        self.up_3 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2,
                                       kernel_size=2, stride=2)
        self.up_3_conv = Modules.DoubleConv3D(base_channels * 4, base_channels * 2)
        self.up_4 = nn.ConvTranspose3d(base_channels * 2, base_channels,
                                       kernel_size=2, stride=2)
        self.up_4_conv = Modules.DoubleConv3D(base_channels * 2, base_channels)

        self.up_5 = nn.ConvTranspose3d(base_channels, base_channels,
                                       kernel_size=2, stride=2)
        self.out = nn.Sequential(nn.Conv3d(base_channels, 2, kernel_size=7, padding=3, padding_mode='replicate'),
                                 nn.BatchNorm3d(2, track_running_stats=False),
                                 nn.ReLU())
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        x_1 = self.conv1(input)
        if self.bn:
            x_1 = self.bn(x_1)
        x_1 = self.relu(x_1)

        x_2 = self.pool(x_1)
        x_2 = self.conv2_1(x_2)
        x_2 = self.conv2_2(x_2)
        x_2 = self.conv2_3(x_2)

        x_3 = self.conv3_1(x_2)
        x_3 = self.conv3_2(x_3)
        x_3 = self.conv3_3(x_3)
        x_3 = self.conv3_4(x_3)

        x_4 = self.conv4_1(x_3)
        x_4 = self.conv4_2(x_4)
        x_4 = self.conv4_3(x_4)
        x_4 = self.conv4_4(x_4)
        x_4 = self.conv4_5(x_4)
        x_4 = self.conv4_6(x_4)

        x_5 = self.conv5_1(x_4)
        x_5 = self.conv5_2(x_5)
        x_5 = self.conv5_3(x_5)

        x = self.up_1(x_5)
        x = torch.cat([x, x_4], dim=1)
        x = self.up_1_conv(x)

        x = self.up_2(x)
        x = torch.cat([x, x_3], dim=1)
        x = self.up_2_conv(x)

        x = self.up_3(x)
        x = torch.cat([x, x_2], dim=1)
        x = self.up_3_conv(x)

        x = self.up_4(x)
        x = torch.cat([x, x_1], dim=1)
        x = self.up_4_conv(x)

        x = self.up_5(x)
        x = self.out(x)
        output = self.final(x)
        return output
