import torch
import torch.nn as nn
from .Modules import FusionNets_Components as FusionModules


# Quan, T. M., Hildebrand, D. G. C., & Jeong, W.-K. (2016) Fusionnet: A deep fully residual convolutional neural network for image segmentation in connectomics. ArXiv Preprint ArXiv:161205360.
# Similar to ResUNet. Except that use addition than concatenation. Which can resolves some issues including gradient vanishing.
# chained FusionNet - few fusionnets chained together. but unlike RNN, the weights of each unit update independently.

# Chained FusionNets (as proposed in paper) is under development.
class SingleFusion(nn.Module):
    def __init__(self, base_channels=64, bn=True):
        super(SingleFusion, self).__init__()
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.encode1 = FusionModules.FusionBlock(1, base_channels, bn=bn)
        self.encode2 = FusionModules.FusionBlock(base_channels, base_channels * 2, bn=bn)
        self.encode3 = FusionModules.FusionBlock(base_channels * 2, base_channels * 4, bn=bn)
        self.encode4 = FusionModules.FusionBlock(base_channels * 4, base_channels * 8, bn=bn)
        self.encode5 = FusionModules.FusionBlock(base_channels * 8, base_channels * 16, bn=bn)

        self.unpool4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 2, 2)
        self.decode4 = FusionModules.FusionBlock(base_channels * 8, base_channels * 8, bn=bn)
        self.unpool3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, 2)
        self.decode3 = FusionModules.FusionBlock(base_channels * 4, base_channels * 4, bn=bn)
        self.unpool2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, 2)
        self.decode2 = FusionModules.FusionBlock(base_channels * 2, base_channels * 2, bn=bn)
        self.unpool1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, 2)
        self.decode1 = FusionModules.FusionBlock(base_channels, base_channels, bn=bn)

        self.final = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        x1 = self.encode1(input)
        x2 = self.pool(x1)
        x2 = self.encode2(x2)
        x3 = self.pool(x2)
        x3 = self.encode3(x3)
        x4 = self.pool(x3)
        x4 = self.encode4(x4)
        x5 = self.pool(x4)
        x5 = self.encode5(x5)

        x = self.unpool4(x5)
        x = x + x4
        x = self.decode4(x)
        x = self.unpool3(x)
        x = x + x3
        x = self.decode3(x)
        x = self.unpool2(x)
        x = x + x2
        x = self.decode2(x)
        x = self.unpool1(x)
        x = x + x1
        x = self.decode1(x)

        output = self.final(x)
        return output
