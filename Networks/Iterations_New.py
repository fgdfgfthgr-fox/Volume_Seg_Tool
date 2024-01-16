import math

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, padding_mode='replicate')
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# A basic 3D-UNet structure.
# Have only 3 max pooling, down-scale the input image to 1/8 due to we are using a volume of only 128x128x128 at a time.
# Compare to the 572x572 input size in the original UNet paper.
class Initial(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Dropouts

# Adding dropout in the middle layer.
class Dropout_Middle(nn.Module):
    def __init__(self, base_channels=8, dropout_p=0.5):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(dropout_p)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.dropout(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Adding dropout in all the encoder layers.
class Dropout_All(nn.Module):
    def __init__(self, base_channels=8, dropout_p=0.5):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(dropout_p)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)
        x1 = self.dropout(x1)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)
        x2 = self.dropout(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)
        x3 = self.dropout(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.dropout(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Adding dropout in all the encoder layers. With a lower dropout probability.
class Dropout_All_Low(nn.Module):
    def __init__(self, base_channels=8, dropout_p=0.25):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(dropout_p)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)
        x1 = self.dropout(x1)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)
        x2 = self.dropout(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)
        x3 = self.dropout(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.dropout(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Instance Normalisation

# DoubleConv with Batch Normalisation changed to Instance Normalisation.
class DoubleConv_IB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, padding_mode='replicate')
        self.bn = nn.InstanceNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# All Batch Normalisation changed to Instance Normalisation.
class InstanceNorm(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv_IB(1, 1 * base_channels)
        self.encode2 = DoubleConv_IB(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv_IB(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv_IB(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv_IB(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv_IB(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv_IB(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Residual Connections

class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3)):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2,
                   (kernel_size[1] - 1) // 2,
                   (kernel_size[2] - 1) // 2)
        if in_channels != out_channels:
            self.mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.mapping = None
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros')
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros')
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_id = x
        x = self.conv_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_2(x)
        if self.mapping:
            x_id = self.mapping(x_id)
        x = x + x_id
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, neck=4):
        super().__init__()
        if in_channels != out_channels:
            self.mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.mapping = None
        middle_channel = math.ceil(out_channels/neck)
        self.conv_1 = nn.Conv3d(in_channels, middle_channel, kernel_size=1)
        self.conv_2 = nn.Conv3d(middle_channel, middle_channel, kernel_size=3,
                                padding=1, padding_mode='zeros')
        self.conv_3 = nn.Conv3d(middle_channel, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(middle_channel, track_running_stats=False)
        self.bn2 = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_id = x
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_3(x)
        if self.mapping:
            x_id = self.mapping(x_id)
        x = x + x_id
        x = self.bn2(x)
        x = self.relu(x)
        return x


# Change all DoubleConv to ResBasicBlock.
class Residual_Basic(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = ResBasicBlock(1, 1 * base_channels)
        self.encode2 = ResBasicBlock(base_channels, 2 * base_channels)
        self.encode3 = ResBasicBlock(2 * base_channels, 4 * base_channels)

        self.middle = ResBasicBlock(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = ResBasicBlock(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = ResBasicBlock(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = ResBasicBlock(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Change all DoubleConv to ResBottleneckBlock.
class Residual_Bottleneck(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = ResBottleneckBlock(1, 1 * base_channels)
        self.encode2 = ResBottleneckBlock(base_channels, 2 * base_channels)
        self.encode3 = ResBottleneckBlock(2 * base_channels, 4 * base_channels)

        self.middle = ResBottleneckBlock(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = ResBottleneckBlock(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = ResBottleneckBlock(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = ResBottleneckBlock(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Change all DoubleConv to ResBottleneckBlock, with neck factor set to 3 instead of the default 4.
class Residual_Bottleneck_Small_Neck(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = ResBottleneckBlock(1, 1 * base_channels, 3)
        self.encode2 = ResBottleneckBlock(base_channels, 2 * base_channels, 3)
        self.encode3 = ResBottleneckBlock(2 * base_channels, 4 * base_channels, 3)

        self.middle = ResBottleneckBlock(4 * base_channels, 8 * base_channels, 3)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = ResBottleneckBlock(8 * base_channels, 4 * base_channels, 3)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = ResBottleneckBlock(4 * base_channels, 2 * base_channels, 3)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = ResBottleneckBlock(2 * base_channels, 1 * base_channels, 3)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Grouped Convolutions

class DoubleConv_Grouped(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality=2):
        super().__init__()
        if in_channels == 1 or in_channels % cardinality != 0:
            in_cardinality = 1
        else:
            in_cardinality = cardinality
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, groups=in_cardinality,
                               kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, groups=cardinality,
                               kernel_size=3, padding=1, padding_mode='replicate')
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Change all Conv3D in the DoubleConv to grouped convolution (aka cardinality)
class GroupedConv(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv_Grouped(1, 1 * base_channels)
        self.encode2 = DoubleConv_Grouped(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv_Grouped(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv_Grouped(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv_Grouped(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv_Grouped(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv_Grouped(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Change all Conv3D in the DoubleConv to grouped convolution (aka cardinality)
# With the cardinality set to 4 instead of 2
class GroupedConv_More(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv_Grouped(1, 1 * base_channels, 4)
        self.encode2 = DoubleConv_Grouped(base_channels, 2 * base_channels, 4)
        self.encode3 = DoubleConv_Grouped(2 * base_channels, 4 * base_channels, 4)

        self.middle = DoubleConv_Grouped(4 * base_channels, 8 * base_channels, 4)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv_Grouped(8 * base_channels, 4 * base_channels, 4)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv_Grouped(4 * base_channels, 2 * base_channels, 4)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv_Grouped(2 * base_channels, 1 * base_channels, 4)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv3d(out_channels // 2, out_channels // 2,
                      kernel_size=3, padding=1, padding_mode='replicate',
                      groups=out_channels // 2),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat((x1, x2), dim=1)


class GhostDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GhostModule(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = GhostModule(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Changed all DoubleConv to GhostDoubleConv
class GhostModules(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = GhostDoubleConv(1, 1 * base_channels)
        self.encode2 = GhostDoubleConv(base_channels, 2 * base_channels)
        self.encode3 = GhostDoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = GhostDoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = GhostDoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = GhostDoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = GhostDoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Squeeze and Excitation Blocks

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_avg = self.fc1(x_avg)
        x_avg = self.relu(x_avg)
        x_avg = self.fc2(x_avg)
        x_avg = self.sigmoid(x_avg)
        return x * x_avg


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
        cse_output = self.cse(x)
        sse_output = self.sse(x)
        return cse_output + sse_output


# Adding a scSE block after every DoubleConv.
class Squeeze_AND_Excite(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.scse1 = scSE(1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.scse2 = scSE(2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)
        self.scse3 = scSE(4 * base_channels)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.scse4 = scSE(8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.scse5 = scSE(4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.scse6 = scSE(2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)
        self.scse7 = scSE(1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)
        x1 = self.scse1(x1)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)
        x2 = self.scse2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)
        x3 = self.scse3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.scse4(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.scse5(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.scse6(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)
        x = self.scse7(x)

        output = self.out(x)
        return output


# Asymmetric Convolutions

class AsyDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, padding_mode='replicate')

        self.conv2_0 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=(1, 1, 3), padding=(0, 0, 1), padding_mode='replicate')
        self.conv2_1 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=(1, 3, 1), padding=(0, 1, 0), padding_mode='replicate')
        self.conv2_2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=(3, 1, 1), padding=(1, 0, 0), padding_mode='replicate')
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Replaced the DoubleConv in the middle layer with AsyDoubleConv.
class Asymmetric(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = AsyDoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Replaced the DoubleConv in the middle layer with AsyDoubleConv. As well as encode3 and decode 3.
class AsymmetricMore(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = AsyDoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = AsyDoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = AsyDoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Pixel Shuffles & Subpixel Convolution

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


class SubpixelConv(nn.Module):
    def __init__(self, in_channels, pre_shuffle_mult=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels*pre_shuffle_mult, kernel_size=1)
        self.shuffle = PixelShuffle3d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffle(x)
        return x


# Use SubpixelConv instead of ConvTranspose.
class SubpixelConvolutions(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = SubpixelConv(8 * base_channels)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = SubpixelConv(4 * base_channels)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = SubpixelConv(2 * base_channels)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Simplified Decoders

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, padding_mode='replicate')
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Merge(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        super().__init__()
        self.scaleup = torch.nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True)
        if in_channels != out_channels:
            self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = None

    def forward(self, x1, x2):
        if self.conv:
            x1 = self.conv(x1)
        x1 = self.scaleup(x1)
        return torch.add(x1, x2)


# All the decoder block have only 1 convolution instead of 2.
class SimplifiedDecoder(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = SingleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = SingleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = SingleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Extremely simplified Decoder design like in Half U-Net.
class HalfUNetDecoder(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)

        self.merge3 = Merge(8, 8 * base_channels, 1 * base_channels)
        self.merge2 = Merge(4, 4 * base_channels, 1 * base_channels)
        self.merge1 = Merge(2, 2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)

        x = self.merge3(x4, x1)
        x = self.merge2(x3, x)
        x = self.merge1(x2, x)

        output = self.out(x)
        return output


# Strided convolutions for down sampling


# All the max pooling layers has been replaced with strided convolutions of kernel size 2.
class StridedConvDown(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.pool1 = nn.Conv3d(1 * base_channels, 1 * base_channels, 2, 2)
        self.encode2 = DoubleConv(1 * base_channels, 2 * base_channels)
        self.pool2 = nn.Conv3d(2 * base_channels, 2 * base_channels, 2, 2)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)
        self.pool3 = nn.Conv3d(4 * base_channels, 4 * base_channels, 2, 2)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.pool1(x1)
        x2 = self.encode2(x2)

        x3 = self.pool2(x2)
        x3 = self.encode3(x3)

        x4 = self.pool3(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# All the max pooling layers has been replaced with strided convolutions of kernel size 3.
class StridedConvDownKS3(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.pad = nn.ReplicationPad3d((0, 1, 0, 1, 0, 1))

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.pool1 = nn.Conv3d(1 * base_channels, 1 * base_channels, 2, 2)
        self.encode2 = DoubleConv(1 * base_channels, 2 * base_channels)
        self.pool2 = nn.Conv3d(2 * base_channels, 2 * base_channels, 2, 2)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)
        self.pool3 = nn.Conv3d(4 * base_channels, 4 * base_channels, 2, 2)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.pool1(self.pad(x1))
        x2 = self.encode2(x2)

        x3 = self.pool2(self.pad(x2))
        x3 = self.encode3(x3)

        x4 = self.pool3(self.pad(x3))
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Double the number of channel first before pooling in order to avoid representational bottleneck.
class PrePoolDouble(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.pre_pool1 = nn.Conv3d(1 * base_channels, 2 * base_channels, 1)
        self.encode2 = DoubleConv(2 * base_channels, 2 * base_channels)
        self.pre_pool2 = nn.Conv3d(2 * base_channels, 4 * base_channels, 1)
        self.encode3 = DoubleConv(4 * base_channels, 4 * base_channels)
        self.pre_pool3 = nn.Conv3d(4 * base_channels, 8 * base_channels, 1)

        self.middle = DoubleConv(8 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.pre_pool1(x1)
        x2 = self.max_pool(x2)
        x2 = self.encode2(x2)

        x3 = self.pre_pool2(x2)
        x3 = self.max_pool(x3)
        x3 = self.encode3(x3)

        x4 = self.pre_pool3(x3)
        x4 = self.max_pool(x4)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Chained Networks

# Two UNet chained together.
class Chained2(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConv(1, 1 * base_channels)
        self.encode2 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

        self.encode1_1 = DoubleConv(1, 1 * base_channels)
        self.encode2_1 = DoubleConv(base_channels, 2 * base_channels)
        self.encode3_1 = DoubleConv(2 * base_channels, 4 * base_channels)

        self.middle_1 = DoubleConv(4 * base_channels, 8 * base_channels)
        self.up3_1 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3_1 = DoubleConv(8 * base_channels, 4 * base_channels)
        self.up2_1 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2_1 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1_1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1_1 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out_1 = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        input = self.out(x)

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Small Non-Residual Net with Residual Net

# A small non-residual UNet and a residual net chained together.
class NonResPreProcess(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1_0 = DoubleConv(1, 1 * base_channels)
        self.encode2_0 = DoubleConv(base_channels, 2 * base_channels)

        self.middle_0 = DoubleConv(2 * base_channels, 4 * base_channels)
        self.up2_0 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)

        self.decode2_0 = DoubleConv(4 * base_channels, 2 * base_channels)
        self.up1_0 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1_0 = DoubleConv(2 * base_channels, 1 * base_channels)

        self.out_0 = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())

        self.encode1 = ResBottleneckBlock(1, 1 * base_channels)
        self.encode2 = ResBottleneckBlock(base_channels, 2 * base_channels)
        self.encode3 = ResBottleneckBlock(2 * base_channels, 4 * base_channels)

        self.middle = ResBottleneckBlock(4 * base_channels, 8 * base_channels)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = ResBottleneckBlock(8 * base_channels, 4 * base_channels)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = ResBottleneckBlock(4 * base_channels, 2 * base_channels)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = ResBottleneckBlock(2 * base_channels, 1 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1_0(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2_0(x2)

        x3 = self.max_pool(x2)
        x3 = self.middle_0(x3)
        x3 = self.up2_0(x3)

        x = torch.cat((x3, x2), dim=1)
        x = self.decode2_0(x)
        x = self.up1_0(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1_0(x)

        pre_processed_input = self.out_0(x)

        x1 = self.encode1(pre_processed_input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Various Different Activation Functions

class DoubleConvDiffAct(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, padding_mode='replicate')
        self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        if activation == "ELU":
            self.activation = nn.ELU(inplace=True)
        elif activation == "PReLU":
            self.activation = nn.PReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# Self Explanatory name
class InitialWithELU(nn.Module):
    def __init__(self, base_channels=8, activation="ELU"):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConvDiffAct(1, 1 * base_channels, activation)
        self.encode2 = DoubleConvDiffAct(base_channels, 2 * base_channels, activation)
        self.encode3 = DoubleConvDiffAct(2 * base_channels, 4 * base_channels, activation)

        self.middle = DoubleConvDiffAct(4 * base_channels, 8 * base_channels, activation)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConvDiffAct(8 * base_channels, 4 * base_channels, activation)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConvDiffAct(4 * base_channels, 2 * base_channels, activation)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConvDiffAct(2 * base_channels, 1 * base_channels, activation)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Self Explanatory name
class InitialWithPReLU(nn.Module):
    def __init__(self, base_channels=8, activation="PReLU"):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConvDiffAct(1, 1 * base_channels, activation)
        self.encode2 = DoubleConvDiffAct(base_channels, 2 * base_channels, activation)
        self.encode3 = DoubleConvDiffAct(2 * base_channels, 4 * base_channels, activation)

        self.middle = DoubleConvDiffAct(4 * base_channels, 8 * base_channels, activation)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConvDiffAct(8 * base_channels, 4 * base_channels, activation)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConvDiffAct(4 * base_channels, 2 * base_channels, activation)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConvDiffAct(2 * base_channels, 1 * base_channels, activation)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output


# Self Explanatory name
class InitialWithGELU(nn.Module):
    def __init__(self, base_channels=8, activation="GELU"):
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)

        self.encode1 = DoubleConvDiffAct(1, 1 * base_channels, activation)
        self.encode2 = DoubleConvDiffAct(base_channels, 2 * base_channels, activation)
        self.encode3 = DoubleConvDiffAct(2 * base_channels, 4 * base_channels, activation)

        self.middle = DoubleConvDiffAct(4 * base_channels, 8 * base_channels, activation)
        self.up3 = nn.LazyConvTranspose3d(4 * base_channels, 2, 2)

        self.decode3 = DoubleConvDiffAct(8 * base_channels, 4 * base_channels, activation)
        self.up2 = nn.LazyConvTranspose3d(2 * base_channels, 2, 2)
        self.decode2 = DoubleConvDiffAct(4 * base_channels, 2 * base_channels, activation)
        self.up1 = nn.LazyConvTranspose3d(1 * base_channels, 2, 2)
        self.decode1 = DoubleConvDiffAct(2 * base_channels, 1 * base_channels, activation)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):

        x1 = self.encode1(input)

        x2 = self.max_pool(x1)
        x2 = self.encode2(x2)

        x3 = self.max_pool(x2)
        x3 = self.encode3(x3)

        x4 = self.max_pool(x3)
        x4 = self.middle(x4)
        x4 = self.up3(x4)

        x = torch.cat((x4, x3), dim=1)
        x = self.decode3(x)
        x = self.up2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.decode2(x)
        x = self.up1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.decode1(x)

        output = self.out(x)
        return output