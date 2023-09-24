import torch
import torch.nn as nn
from .Modules import SegNets_Components as SegModules


# V. et al.  (2017) Segnet: a deep convolutional encoder-decoder architecture for image segmentation. IEEE Trans. Pattern Anal. Mach. Intell., 39, 2481–2495.

# Use max pooling indices to restore spatial information, hence no skip connections used.

# What I implemented here is actually "SegNet-Basic" in the paper.
# Also, the default kernel size for all conv layers is 3 instead of 7 (in the paper) due to the size of 7 take too much resources in the 3D cases.

class SegNet(nn.Module):
    def __init__(self, base_channels=64, kernel_size=(3, 3, 3)):
        super(SegNet, self).__init__()
        self.encode1 = SegModules.SegEncodeBlock_2(1, base_channels, conv_kernel_size=kernel_size)
        self.encode2 = SegModules.SegEncodeBlock_2(base_channels, base_channels * 2, conv_kernel_size=kernel_size)
        self.encode3 = SegModules.SegEncodeBlock_3(base_channels * 2, base_channels * 4, conv_kernel_size=kernel_size)
        self.encode4 = SegModules.SegEncodeBlock_3(base_channels * 4, base_channels * 8, conv_kernel_size=kernel_size)
        self.encode5 = SegModules.SegEncodeBlock_3(base_channels * 8, base_channels * 16, conv_kernel_size=kernel_size)

        self.decode1 = SegModules.SegDecodeBlock_3(base_channels * 16, base_channels * 8, conv_kernel_size=kernel_size)
        self.decode2 = SegModules.SegDecodeBlock_3(base_channels * 8, base_channels * 4, conv_kernel_size=kernel_size)
        self.decode3 = SegModules.SegDecodeBlock_3(base_channels * 4, base_channels * 2, conv_kernel_size=kernel_size)
        self.decode4 = SegModules.SegDecodeBlock_2(base_channels * 2, base_channels, conv_kernel_size=kernel_size)
        self.decode5 = SegModules.SegDecodeBlock_2(base_channels, 2, conv_kernel_size=kernel_size)
        # modified output layer to suit the binary classification task
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        x, indice1 = self.encode1(input)
        x, indice2 = self.encode2(x)
        x, indice3 = self.encode3(x)
        x, indice4 = self.encode4(x)
        x, indice5 = self.encode5(x)

        x = self.decode1(x, indice5)
        x = self.decode2(x, indice4)
        x = self.decode3(x, indice3)
        x = self.decode4(x, indice2)
        x = self.decode5(x, indice1)

        output = self.final(x)
        return output