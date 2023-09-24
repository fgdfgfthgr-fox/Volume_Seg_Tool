import torch
import torch.nn as nn
from .Modules import HighRes3DNet_Components as Modules


# W. Li, G. Wang, L. Fidon, S. Ourselin, M.J. Cardoso, T. Vercauteren, On the compactness, efficiency,
# and representation of 3D convolutional networks: brain parcellation as a pretext task, in: International Conference
# on Information Processing in Medical Imaging, Springer, 2017, pp. 348–360, doi:10.1007/978- 3- 319- 59050- 9_28.

# Utilised dilated convolutions and residual connections.
# The spatial resolution of the input volume is maintained throughout the network.

# The original paper used a 160-way softmax classifier as the last layer due to its 160 class classification task
class HighRes3DNets(nn.Module):
    def __init__(self, base_channels=16, bn=True):
        super(HighRes3DNets, self).__init__()

        self.inc = nn.Conv3d(1, base_channels, kernel_size=3, padding=1, padding_mode="replicate")
        if bn:
            self.bn = nn.BatchNorm3d(base_channels, track_running_stats=False)
        self.relu = nn.ReLU()

        self.block_1_1 = Modules.HighRes3DNetBlock(base_channels, base_channels, bn=bn)
        self.block_1_2 = Modules.HighRes3DNetBlock(base_channels, base_channels, bn=bn)
        self.block_1_3 = Modules.HighRes3DNetBlock(base_channels, base_channels, bn=bn)

        self.block_2_1 = Modules.HighRes3DNetBlock(base_channels, 2 * base_channels, dilation=(2, 2, 2), bn=bn)
        self.block_2_2 = Modules.HighRes3DNetBlock(2 * base_channels, 2 * base_channels, dilation=(2, 2, 2), bn=bn)
        self.block_2_3 = Modules.HighRes3DNetBlock(2 * base_channels, 2 * base_channels, dilation=(2, 2, 2), bn=bn)

        self.block_3_1 = Modules.HighRes3DNetBlock(2 * base_channels, 4 * base_channels, dilation=(4, 4, 4), bn=bn)
        self.block_3_2 = Modules.HighRes3DNetBlock(4 * base_channels, 4 * base_channels, dilation=(4, 4, 4), bn=bn)
        self.block_3_3 = Modules.HighRes3DNetBlock(4 * base_channels, 4 * base_channels, dilation=(4, 4, 4), bn=bn)

        self.out = nn.Sequential(nn.Conv3d(4 * base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        x = self.inc(input)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)

        x = self.block_1_1(x)
        x = self.block_1_2(x)
        x = self.block_1_3(x)

        x = self.block_2_1(x)
        x = self.block_2_2(x)
        x = self.block_2_3(x)

        x = self.block_3_1(x)
        x = self.block_3_2(x)
        x = self.block_3_3(x)

        output = self.out(x)
        return output