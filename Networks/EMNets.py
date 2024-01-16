import torch
import torch.nn as nn
from .Modules import EMNets_Components as EMModules
from .Modules import General_Components as Modules


# Khadangi,A. et al. (2020) EM-net: deep learning for electron microscopy image segmentation. bioRxiv.
# Uses Trainable Linear Unit (TLU) and spatial dropout layers. Light weight with low computational complexity.


class EMNet_V1_BN(nn.Module):
    def __init__(self, base_channels=16):
        super(EMNet_V1_BN, self).__init__()
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(p=0.5)

        self.encode1_1 = EMModules.EMNetCell1_3D(1, base_channels, bn=True)
        self.encode1_2 = EMModules.EMNetCell1_3D(base_channels, base_channels, bn=True)

        self.encode2_1 = EMModules.EMNetCell1_3D(2 * base_channels, 2 * base_channels, bn=True)
        self.encode2_2 = EMModules.EMNetCell1_3D(2 * base_channels, 2 * base_channels, bn=True)

        self.encode3_1 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels, bn=True)
        self.encode3_2 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels, bn=True)

        self.encode4_1 = EMModules.EMNetCell1_3D(8 * base_channels, 8 * base_channels, bn=True)
        self.encode4_2 = EMModules.EMNetCell1_3D(8 * base_channels, 8 * base_channels, bn=True)

        self.encode5_1 = EMModules.EMNetCell1_3D(16 * base_channels, 4 * base_channels, bn=True)
        self.encode5_2 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels, bn=True)
        self.encode5_3 = EMModules.EMNetCell1_3D(16 * base_channels, 4 * base_channels, bn=True)
        self.encode5_4 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels, bn=True)

        self.up4 = EMModules.EMNetUp_3D(8 * base_channels, 16 * base_channels)
        self.decode4 = Modules.DoubleConv3D(24 * base_channels, 16 * base_channels)

        self.up3 = EMModules.EMNetUp_3D(16 * base_channels, 8 * base_channels)
        self.decode3 = Modules.DoubleConv3D(12 * base_channels, 8 * base_channels)

        self.up2 = EMModules.EMNetUp_3D(8 * base_channels, 4 * base_channels)
        self.decode2 = Modules.DoubleConv3D(6 * base_channels, 4 * base_channels)

        self.up1 = EMModules.EMNetUp_3D(4 * base_channels, 2 * base_channels)
        self.decode1 = Modules.DoubleConv3D(3 * base_channels, 2 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(2 * base_channels, 2, kernel_size=3, padding=1, padding_mode='replicate'),
                                 nn.BatchNorm3d(2, track_running_stats=False),
                                 nn.ReLU())
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        conv1A = self.encode1_1(input)
        conv1B = self.encode1_2(conv1A)
        pool1 = torch.cat([conv1A, conv1B], dim=1)
        pool1 = self.pool(pool1)

        conv2A = self.encode2_1(pool1)
        conv2B = self.encode2_2(conv2A)
        pool2 = torch.cat([conv2A, conv2B], dim=1)
        pool2 = self.pool(pool2)

        conv3A = self.encode3_1(pool2)
        conv3B = self.encode3_2(conv3A)
        pool3 = torch.cat([conv3A, conv3B], dim=1)
        pool3 = self.pool(pool3)

        conv4A = self.encode4_1(pool3)
        conv4B = self.encode4_2(conv4A)
        pool4 = torch.cat([conv4A, conv4B], dim=1)
        pool4 = self.pool(pool4)
        pool4 = self.dropout(pool4)

        conv5A = self.encode5_1(pool4)
        conv5B = self.encode5_2(conv5A)
        conv5C = self.encode5_3(pool4)
        conv5D = self.encode5_4(conv5C)
        pool5 = torch.cat([conv5B, conv5D], dim=1)

        up6 = self.up4(pool5)
        merge6 = torch.cat([up6, conv4B], dim=1)

        conv6 = self.decode4(merge6)

        up7 = self.up3(conv6)
        merge7 = torch.cat([up7, conv3B], dim=1)

        conv7 = self.decode3(merge7)

        up8 = self.up2(conv7)
        merge8 = torch.cat([up8, conv2B], dim=1)

        conv8 = self.decode2(merge8)

        up9 = self.up1(conv8)
        merge9 = torch.cat([up9, conv1B], dim=1)

        conv9 = self.decode1(merge9)

        conv9 = self.out(conv9)
        result = self.final(conv9)
        return result

class EMNet_V1_BN_2X(nn.Module):
    def __init__(self, base_channels=16):
        super(EMNet_V1_BN_2X, self).__init__()
        self.pool = nn.MaxPool3d(2)

        self.conv1A = EMModules.EMNetCell1_3D(1, base_channels, bn=True)
        self.conv1B = EMModules.EMNetCell1_3D(base_channels, base_channels // 2, bn=True)
        self.conv1C = EMModules.EMNetCell1_3D(base_channels // 2, base_channels, bn=True)
        self.conv1D = EMModules.EMNetCell1_3D(base_channels, base_channels // 2, bn=True)

        self.conv2A = EMModules.EMNetCell1_3D(base_channels, 2 * base_channels, bn=True)
        self.conv2B = EMModules.EMNetCell1_3D(2 * base_channels, base_channels, bn=True)
        self.conv2C = EMModules.EMNetCell1_3D(base_channels, 2 * base_channels, bn=True)
        self.conv2D = EMModules.EMNetCell1_3D(2 * base_channels, base_channels, bn=True)

        self.conv3A = EMModules.EMNetCell1_3D(2 * base_channels, 4 * base_channels, bn=True)
        self.conv3B = EMModules.EMNetCell1_3D(4 * base_channels, 2 * base_channels, bn=True)
        self.conv3C = EMModules.EMNetCell1_3D(2 * base_channels, 4 * base_channels, bn=True)
        self.conv3D = EMModules.EMNetCell1_3D(4 * base_channels, 2 * base_channels, bn=True)

        self.conv4A = EMModules.EMNetCell1_3D(4 * base_channels, 8 * base_channels, bn=True)
        self.conv4B = EMModules.EMNetCell1_3D(8 * base_channels, 4 * base_channels, bn=True)
        self.conv4C = EMModules.EMNetCell1_3D(4 * base_channels, 8 * base_channels, bn=True)
        self.conv4D = EMModules.EMNetCell1_3D(8 * base_channels, 4 * base_channels, bn=True)

        self.conv5A = EMModules.EMNetCell1_3D(8 * base_channels, 8 * base_channels, bn=True)
        self.conv5B = EMModules.EMNetCell1_3D(8 * base_channels, 8 * base_channels, bn=True)
        self.conv5C = EMModules.EMNetCell1_3D(8 * base_channels, 8 * base_channels, bn=True)
        self.conv5D = EMModules.EMNetCell1_3D(8 * base_channels, 8 * base_channels, bn=True)

        self.up4 = EMModules.EMNetUp_3D(16 * base_channels, 16 * base_channels)
        self.decode4 = Modules.DoubleConv3D(20 * base_channels, 16 * base_channels)

        self.up3 = EMModules.EMNetUp_3D(16 * base_channels, 8 * base_channels)
        self.decode3 = Modules.DoubleConv3D(10 * base_channels, 8 * base_channels)

        self.up2 = EMModules.EMNetUp_3D(8 * base_channels, 4 * base_channels)
        self.decode2 = Modules.DoubleConv3D(5 * base_channels, 4 * base_channels)

        self.up1 = EMModules.EMNetUp_3D(4 * base_channels, 2 * base_channels)
        self.decode1 = Modules.DoubleConv3D(int(2.5 * base_channels), 2 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(2 * base_channels, 2, kernel_size=3, padding=1, padding_mode='replicate'),
                                 nn.BatchNorm3d(2, track_running_stats=False),
                                 nn.ReLU())
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        conv1A = self.conv1A(input)
        conv1B = self.conv1B(conv1A)
        conv1C = self.conv1C(conv1B)
        conv1D = self.conv1D(conv1C)
        pool1 = torch.cat([conv1B, conv1D], dim=1)
        pool1 = self.pool(pool1)


        conv2A = self.conv2A(pool1)
        conv2B = self.conv2B(conv2A)
        conv2C = self.conv2C(conv2B)
        conv2D = self.conv2D(conv2C)
        pool2 = torch.cat([conv2B, conv2D], dim=1)
        pool2 = self.pool(pool2)

        conv3A = self.conv3A(pool2)
        conv3B = self.conv3B(conv3A)
        conv3C = self.conv3C(conv3B)
        conv3D = self.conv3D(conv3C)
        pool3 = torch.cat([conv3B, conv3D], dim=1)
        pool3 = self.pool(pool3)

        conv4A = self.conv4A(pool3)
        conv4B = self.conv4B(conv4A)
        conv4C = self.conv4C(conv4B)
        conv4D = self.conv4D(conv4C)
        pool4 = torch.cat([conv4B, conv4D], dim=1)
        pool4 = self.pool(pool4)

        conv5A = self.conv5A(pool4)
        conv5B = self.conv5B(conv5A)
        conv5C = self.conv5C(conv5B)
        conv5D = self.conv5D(conv5C)
        pool5 = torch.cat([conv5B, conv5D], dim=1)

        up6 = self.up4(pool5)
        merge6 = torch.cat([up6, conv4D], dim=1)

        conv6 = self.decode4(merge6)

        up7 = self.up3(conv6)
        merge7 = torch.cat([up7, conv3D], dim=1)

        conv7 = self.decode3(merge7)

        up8 = self.up2(conv7)
        merge8 = torch.cat([up8, conv2D], dim=1)

        conv8 = self.decode2(merge8)

        up9 = self.up1(conv8)
        merge9 = torch.cat([up9, conv1D], dim=1)

        conv9 = self.decode1(merge9)

        conv9 = self.out(conv9)
        result = self.final(conv9)
        return result


class EMNet_V1_2X(nn.Module):
    def __init__(self, base_channels=16):
        super(EMNet_V1_2X, self).__init__()
        self.pool = nn.MaxPool3d(2)
        #self.dropout = nn.Dropout3d(p=0.5)

        self.encode1_1 = EMModules.EMNetCell1_3D(1, base_channels)
        self.encode1_2 = EMModules.EMNetCell2_3D(base_channels, base_channels)

        self.encode2_1 = EMModules.EMNetCell1_3D(2 * base_channels, 2 * base_channels)
        self.encode2_2 = EMModules.EMNetCell2_3D(2 * base_channels, 2 * base_channels)

        self.encode3_1 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels)
        self.encode3_2 = EMModules.EMNetCell2_3D(4 * base_channels, 4 * base_channels)

        self.encode4_1 = EMModules.EMNetCell1_3D(8 * base_channels, 8 * base_channels)
        self.encode4_2 = EMModules.EMNetCell2_3D(8 * base_channels, 8 * base_channels)

        self.encode5_1 = EMModules.EMNetCell1_3D(16 * base_channels, 4 * base_channels)
        self.encode5_2 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels)
        self.encode5_3 = EMModules.EMNetCell2_3D(16 * base_channels, 4 * base_channels)
        self.encode5_4 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels)

        self.up4 = EMModules.EMNetUp_3D(8 * base_channels, 16 * base_channels)
        self.decode4 = Modules.DoubleConv3D(24 * base_channels, 16 * base_channels)

        self.up3 = EMModules.EMNetUp_3D(16 * base_channels, 8 * base_channels)
        self.decode3 = Modules.DoubleConv3D(12 * base_channels, 8 * base_channels)

        self.up2 = EMModules.EMNetUp_3D(8 * base_channels, 4 * base_channels)
        self.decode2 = Modules.DoubleConv3D(6 * base_channels, 4 * base_channels)

        self.up1 = EMModules.EMNetUp_3D(4 * base_channels, 2 * base_channels)
        self.decode1 = Modules.DoubleConv3D(3 * base_channels, 2 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(2 * base_channels, 2, kernel_size=3, padding=1, padding_mode='replicate'),
                                 nn.BatchNorm3d(2, track_running_stats=False),
                                 nn.ReLU())
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        conv1A = self.encode1_1(input)
        conv1B = self.encode1_2(conv1A)
        pool1 = torch.cat([conv1A, conv1B], dim=1)
        pool1 = self.pool(pool1)

        conv2A = self.encode2_1(pool1)
        conv2B = self.encode2_2(conv2A)
        pool2 = torch.cat([conv2A, conv2B], dim=1)
        pool2 = self.pool(pool2)

        conv3A = self.encode3_1(pool2)
        conv3B = self.encode3_2(conv3A)
        pool3 = torch.cat([conv3A, conv3B], dim=1)
        pool3 = self.pool(pool3)

        conv4A = self.encode4_1(pool3)
        conv4B = self.encode4_2(conv4A)
        pool4 = torch.cat([conv4A, conv4B], dim=1)
        pool4 = self.pool(pool4)
        #pool4 = self.dropout(pool4)

        conv5A = self.encode5_1(pool4)
        conv5B = self.encode5_2(conv5A)
        conv5C = self.encode5_3(pool4)
        conv5D = self.encode5_4(conv5C)
        pool5 = torch.cat([conv5B, conv5D], dim=1)

        up6 = self.up4(pool5)
        merge6 = torch.cat([up6, conv4B], dim=1)

        conv6 = self.decode4(merge6)

        up7 = self.up3(conv6)
        merge7 = torch.cat([up7, conv3B], dim=1)

        conv7 = self.decode3(merge7)

        up8 = self.up2(conv7)
        merge8 = torch.cat([up8, conv2B], dim=1)

        conv8 = self.decode2(merge8)

        up9 = self.up1(conv8)
        merge9 = torch.cat([up9, conv1B], dim=1)

        conv9 = self.decode1(merge9)

        conv9 = self.out(conv9)
        result = self.final(conv9)
        return result


class EMNet_V1_4X(nn.Module):
    def __init__(self, base_channels=32):
        super(EMNet_V1_4X, self).__init__()
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(p=0.5)
        self.dropout_2 = nn.Dropout3d(p=0.5)

        self.encode1_1 = EMModules.EMNetCell1_3D(1, base_channels)
        self.encode1_2 = EMModules.EMNetCell2_3D(base_channels, base_channels)

        self.encode2_1 = EMModules.EMNetCell1_3D(2 * base_channels, 2 * base_channels)
        self.encode2_2 = EMModules.EMNetCell2_3D(2 * base_channels, 2 * base_channels)

        self.encode3_1 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels)
        self.encode3_2 = EMModules.EMNetCell2_3D(4 * base_channels, 4 * base_channels)

        self.encode4_1 = EMModules.EMNetCell1_3D(8 * base_channels, 8 * base_channels)
        self.encode4_2 = EMModules.EMNetCell2_3D(8 * base_channels, 8 * base_channels)

        self.encode5_1 = EMModules.EMNetCell1_3D(16 * base_channels, 4 * base_channels)
        self.encode5_2 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels)
        self.encode5_3 = EMModules.EMNetCell2_3D(16 * base_channels, 4 * base_channels)
        self.encode5_4 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels)

        self.up4 = EMModules.EMNetUp_3D(8 * base_channels, 16 * base_channels)
        self.decode4 = Modules.DoubleConv3D(24 * base_channels, 16 * base_channels)

        self.up3 = EMModules.EMNetUp_3D(16 * base_channels, 8 * base_channels)
        self.decode3 = Modules.DoubleConv3D(12 * base_channels, 8 * base_channels)

        self.up2 = EMModules.EMNetUp_3D(8 * base_channels, 4 * base_channels)
        self.decode2 = Modules.DoubleConv3D(6 * base_channels, 4 * base_channels)

        self.up1 = EMModules.EMNetUp_3D(4 * base_channels, 2 * base_channels)
        self.decode1 = Modules.DoubleConv3D(3 * base_channels, 2 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(2 * base_channels, 2, kernel_size=3, padding=1, padding_mode='replicate'),
                                 nn.BatchNorm3d(2, track_running_stats=False),
                                 nn.ReLU())
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        conv1A = self.encode1_1(input)
        conv1B = self.encode1_2(conv1A)
        pool1 = torch.cat([conv1A, conv1B], dim=1)
        pool1 = self.pool(pool1)

        conv2A = self.encode2_1(pool1)
        conv2B = self.encode2_2(conv2A)
        pool2 = torch.cat([conv2A, conv2B], dim=1)
        pool2 = self.pool(pool2)

        conv3A = self.encode3_1(pool2)
        conv3B = self.encode3_2(conv3A)
        pool3 = torch.cat([conv3A, conv3B], dim=1)
        pool3 = self.pool(pool3)
        pool3 = self.dropout(pool3)

        conv4A = self.encode4_1(pool3)
        conv4B = self.encode4_2(conv4A)
        pool4 = torch.cat([conv4A, conv4B], dim=1)
        pool4 = self.pool(pool4)
        pool4 = self.dropout_2(pool4)

        conv5A = self.encode5_1(pool4)
        conv5B = self.encode5_2(conv5A)
        conv5C = self.encode5_3(pool4)
        conv5D = self.encode5_4(conv5C)
        pool5 = torch.cat([conv5B, conv5D], dim=1)

        up6 = self.up4(pool5)
        merge6 = torch.cat([up6, conv4B], dim=1)

        conv6 = self.decode4(merge6)

        up7 = self.up3(conv6)
        merge7 = torch.cat([up7, conv3B], dim=1)

        conv7 = self.decode3(merge7)

        up8 = self.up2(conv7)
        merge8 = torch.cat([up8, conv2B], dim=1)

        conv8 = self.decode2(merge8)

        up9 = self.up1(conv8)
        merge9 = torch.cat([up9, conv1B], dim=1)

        conv9 = self.decode1(merge9)

        conv9 = self.out(conv9)
        result = self.final(conv9)
        return result


class EMNet_V2(nn.Module):
    def __init__(self, base_channels=16):
        super(EMNet_V2, self).__init__()
        self.pool = nn.MaxPool3d(2)

        self.encode1_1 = EMModules.EMNetCell2_3D(1, base_channels)
        self.encode1_2 = EMModules.EMNetCell2_3D(base_channels, base_channels)

        self.encode2_1 = EMModules.EMNetCell2_3D(2 * base_channels, 2 * base_channels)
        self.encode2_2 = EMModules.EMNetCell2_3D(2 * base_channels, 2 * base_channels)

        self.encode3_1 = EMModules.EMNetCell2_3D(4 * base_channels, 4 * base_channels)
        self.encode3_2 = EMModules.EMNetCell2_3D(4 * base_channels, 4 * base_channels)

        self.encode4_1 = EMModules.EMNetCell2_3D(8 * base_channels, 8 * base_channels)
        self.encode4_2 = EMModules.EMNetCell2_3D(8 * base_channels, 8 * base_channels)

        self.encode5_1 = EMModules.EMNetCell1_3D(16 * base_channels, 4 * base_channels)
        self.encode5_2 = EMModules.EMNetCell2_3D(4 * base_channels, 4 * base_channels)
        self.encode5_3 = EMModules.EMNetCell2_3D(16 * base_channels, 4 * base_channels)
        self.encode5_4 = EMModules.EMNetCell1_3D(4 * base_channels, 4 * base_channels)

        self.up4 = EMModules.EMNetUp_3D(8 * base_channels, 16 * base_channels)
        self.decode4 = Modules.DoubleConv3D(24 * base_channels, 16 * base_channels)

        self.up3 = EMModules.EMNetUp_3D(16 * base_channels, 8 * base_channels)
        self.decode3 = Modules.DoubleConv3D(12 * base_channels, 8 * base_channels)

        self.up2 = EMModules.EMNetUp_3D(8 * base_channels, 4 * base_channels)
        self.decode2 = Modules.DoubleConv3D(6 * base_channels, 4 * base_channels)

        self.up1 = EMModules.EMNetUp_3D(4 * base_channels, 2 * base_channels)
        self.decode1 = Modules.DoubleConv3D(3 * base_channels, 2 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(2 * base_channels, 2, kernel_size=3, padding=1, padding_mode='replicate'),
                                 nn.BatchNorm3d(2, track_running_stats=False),
                                 nn.ReLU())
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        conv1A = self.encode1_1(input)
        conv1B = self.encode1_2(conv1A)
        pool1 = torch.cat([conv1A, conv1B], dim=1)
        pool1 = self.pool(pool1)

        conv2A = self.encode2_1(pool1)
        conv2B = self.encode2_2(conv2A)
        pool2 = torch.cat([conv2A, conv2B], dim=1)
        pool2 = self.pool(pool2)

        conv3A = self.encode3_1(pool2)
        conv3B = self.encode3_2(conv3A)
        pool3 = torch.cat([conv3A, conv3B], dim=1)
        pool3 = self.pool(pool3)

        conv4A = self.encode4_1(pool3)
        conv4B = self.encode4_2(conv4A)
        pool4 = torch.cat([conv4A, conv4B], dim=1)
        pool4 = self.pool(pool4)

        conv5A = self.encode5_1(pool4)
        conv5B = self.encode5_2(conv5A)
        conv5C = self.encode5_3(pool4)
        conv5D = self.encode5_4(conv5C)
        pool5 = torch.cat([conv5B, conv5D], dim=1)

        up6 = self.up4(pool5)
        merge6 = torch.cat([up6, conv4B], dim=1)

        conv6 = self.decode4(merge6)

        up7 = self.up3(conv6)
        merge7 = torch.cat([up7, conv3B], dim=1)

        conv7 = self.decode3(merge7)

        up8 = self.up2(conv7)
        merge8 = torch.cat([up8, conv2B], dim=1)

        conv8 = self.decode2(merge8)

        up9 = self.up1(conv8)
        merge9 = torch.cat([up9, conv1B], dim=1)

        conv9 = self.decode1(merge9)

        conv9 = self.out(conv9)
        result = self.final(conv9)
        return result


class EMNet_V2_2X(nn.Module):
    def __init__(self, base_channels=16):
        super(EMNet_V2_2X, self).__init__()
        self.pool = nn.MaxPool3d(2)

        self.encode1_1 = EMModules.EMNetCell2_3D(1, base_channels)
        self.encode1_2 = EMModules.EMNetCell2_3D(base_channels, 2 * base_channels)

        self.encode2_1 = EMModules.EMNetCell2_3D(2 * base_channels, 2 * base_channels)
        self.encode2_2 = EMModules.EMNetCell2_3D(2 * base_channels, 4 * base_channels)

        self.encode3_1 = EMModules.EMNetCell2_3D(4 * base_channels, 4 * base_channels)
        self.encode3_2 = EMModules.EMNetCell2_3D(4 * base_channels, 8 * base_channels)

        self.encode4_1 = EMModules.EMNetCell2_3D(8 * base_channels, 8 * base_channels)
        self.encode4_2 = EMModules.EMNetCell2_3D(8 * base_channels, 16 * base_channels)

        self.encode5_1 = EMModules.EMNetCell1_3D(16 * base_channels, 4 * base_channels)
        self.encode5_2 = EMModules.EMNetCell2_3D(4 * base_channels, 8 * base_channels)
        self.encode5_3 = EMModules.EMNetCell2_3D(16 * base_channels, 4 * base_channels)
        self.encode5_4 = EMModules.EMNetCell1_3D(4 * base_channels, 8 * base_channels)

        self.up4 = EMModules.EMNetUp_3D(16 * base_channels, 16 * base_channels)
        self.decode4 = Modules.DoubleConv3D(32 * base_channels, 16 * base_channels)

        self.up3 = EMModules.EMNetUp_3D(16 * base_channels, 8 * base_channels)
        self.decode3 = Modules.DoubleConv3D(16 * base_channels, 8 * base_channels)

        self.up2 = EMModules.EMNetUp_3D(8 * base_channels, 4 * base_channels)
        self.decode2 = Modules.DoubleConv3D(8 * base_channels, 4 * base_channels)

        self.up1 = EMModules.EMNetUp_3D(4 * base_channels, 2 * base_channels)
        self.decode1 = Modules.DoubleConv3D(4 * base_channels, 2 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(2 * base_channels, 2, kernel_size=3, padding=1, padding_mode='replicate'),
                                 nn.BatchNorm3d(2, track_running_stats=False),
                                 nn.ReLU())
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        conv1A = self.encode1_1(input)
        conv1B = self.encode1_2(conv1A)
        pool1 = self.pool(conv1B)

        conv2A = self.encode2_1(pool1)
        conv2B = self.encode2_2(conv2A)
        pool2 = self.pool(conv2B)

        conv3A = self.encode3_1(pool2)
        conv3B = self.encode3_2(conv3A)
        pool3 = self.pool(conv3B)

        conv4A = self.encode4_1(pool3)
        conv4B = self.encode4_2(conv4A)
        pool4 = self.pool(conv4B)

        conv5A = self.encode5_1(pool4)
        conv5B = self.encode5_2(conv5A)
        conv5C = self.encode5_3(pool4)
        conv5D = self.encode5_4(conv5C)
        pool5 = torch.cat([conv5B, conv5D], dim=1)

        up6 = self.up4(pool5)
        merge6 = torch.cat([up6, conv4B], dim=1)

        conv6 = self.decode4(merge6)

        up7 = self.up3(conv6)
        merge7 = torch.cat([up7, conv3B], dim=1)

        conv7 = self.decode3(merge7)

        up8 = self.up2(conv7)
        merge8 = torch.cat([up8, conv2B], dim=1)

        conv8 = self.decode2(merge8)

        up9 = self.up1(conv8)
        merge9 = torch.cat([up9, conv1B], dim=1)

        conv9 = self.decode1(merge9)

        conv9 = self.out(conv9)
        result = self.final(conv9)
        return result


class EMNet_V2_4X(nn.Module):
    def __init__(self, base_channels=32):
        super(EMNet_V2_4X, self).__init__()
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(p=0.5)

        self.conv1 = EMModules.EMNetCellV24X_3D(1, base_channels)
        self.conv2 = EMModules.EMNetCellV24X_3D(base_channels * 4, base_channels * 2)
        self.conv3 = EMModules.EMNetCellV24X_3D(base_channels * 8, base_channels * 4)
        self.conv4 = EMModules.EMNetCellV24X_3D(base_channels * 16, base_channels * 8)
        self.conv5 = EMModules.EMNetCellV24X_3D_MID(base_channels * 32, base_channels * 32)

        self.up4 = EMModules.EMNetUp_3D(32 * base_channels, 32 * base_channels)
        self.decode4 = Modules.DoubleConv3D(64 * base_channels, 32 * base_channels)

        self.up3 = EMModules.EMNetUp_3D(32 * base_channels, 16 * base_channels)
        self.decode3 = Modules.DoubleConv3D(32 * base_channels, 16 * base_channels)

        self.up2 = EMModules.EMNetUp_3D(16 * base_channels, 8 * base_channels)
        self.decode2 = Modules.DoubleConv3D(16 * base_channels, 8 * base_channels)

        self.up1 = EMModules.EMNetUp_3D(8 * base_channels, 4 * base_channels)
        self.decode1 = Modules.DoubleConv3D(8 * base_channels, 4 * base_channels)

        self.out = nn.Sequential(nn.Conv3d(4 * base_channels, 2, kernel_size=3, padding=1, padding_mode='replicate'),
                                 nn.BatchNorm3d(2, track_running_stats=False),
                                 nn.ReLU())
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        conv1A = self.conv1(input)
        pool1 = self.pool(conv1A)

        conv2A = self.conv2(pool1)
        pool2 = self.pool(conv2A)

        conv3A = self.conv3(pool2)
        pool3 = self.pool(conv3A)

        conv4A = self.conv4(pool3)
        pool4 = self.pool(conv4A)

        conv5A = self.conv5(pool4)
        pool5 = self.dropout(conv5A)

        up6 = self.up4(pool5)
        merge6 = torch.cat([up6, conv4A], dim=1)

        conv6 = self.decode4(merge6)

        up7 = self.up3(conv6)
        merge7 = torch.cat([up7, conv3A], dim=1)

        conv7 = self.decode3(merge7)

        up8 = self.up2(conv7)
        merge8 = torch.cat([up8, conv2A], dim=1)

        conv8 = self.decode2(merge8)

        up9 = self.up1(conv8)
        merge9 = torch.cat([up9, conv1A], dim=1)

        conv9 = self.decode1(merge9)

        conv9 = self.out(conv9)
        result = self.final(conv9)
        return result