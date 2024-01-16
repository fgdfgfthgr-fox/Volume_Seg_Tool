import torch
import torch.nn as nn
from .Modules import UNetPPs_Components as PPModules

# Z Zhou, MM Rahman Siddiquee, N Tajbakhsh, et al., in: Unet++: A nested u-net architecture for medical image
# segmentation[M]//Deep learning in medical image analysis and multimodal learning for clinical decision support,
# Springer, Cham, 2018, pp. 3–11.
#
# Copy pasted from: https://github.com/4uiiurz1/pytorch-nested-unet

# nested=connections between the layers are organized in multiple levels, instead of having a single direct connection
# between corresponding layers of the encoder and decoder

# dense=each layer in the decoder is connected to multiple layers in the encoder, allowing for a richer flow of
# information across different levels of abstraction

# underlying hypothesis: network can better segment fine details of the object when high-res feature maps from encoder
# are gradually enriched prior to fusion with the corresponding semantically rich feature maps from the decoder network.


class NestedUNet(nn.Module):
    def __init__(self, base_channels=32, num_classes=1, input_channels=1, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool3d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = PPModules.VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = PPModules.VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = PPModules.VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = PPModules.VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = PPModules.VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = PPModules.VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = PPModules.VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = PPModules.VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = PPModules.VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = PPModules.VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = PPModules.VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = PPModules.VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = PPModules.VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = PPModules.VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = PPModules.VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
        self.out = nn.Sigmoid()


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.out(self.final1(x0_1))
            output2 = self.out(self.final2(x0_2))
            output3 = self.out(self.final3(x0_3))
            output4 = self.out(self.final4(x0_4))
            return [output1, output2, output3, output4]

        else:
            output = self.out(self.final(x0_4))
            return output