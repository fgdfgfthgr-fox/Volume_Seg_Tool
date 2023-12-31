import torch
import torch.nn as nn
from .Modules import General_Components as Modules


# Ronneberger O. et al. (2015) U-net: convolutional networks for biomedical image segmentation. In: International
# Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, pp. 234–241.

# The most influential DL image classification paper. Introduced the concept of skip connection to feed spatial information into the decoder part of the network.
# The original UNet implementation in the paper doesn't use batch normalization, but today BN is commonly used in UNet. Also they used unpadded conv.

class UNet(nn.Module):
    def __init__(self, base_channels=64, bn=True, depth=5):
        super(UNet, self).__init__()
        self.depth = depth
        assert depth >= 3, "The depth needs to be at least 3."

        for i in range(1, depth):
            if i == 1:
                setattr(self, f'encode{i}',
                        Modules.DoubleConv3D(1, base_channels * (2 ** (i - 1)), bn=bn))
                setattr(self, f'decode{i}',
                        Modules.Up3D(base_channels * (2 ** i), base_channels * (2 ** (i - 1)), bn=bn))
            else:
                setattr(self, f'encode{i}', Modules.Down3D(base_channels * (2**(i-2)), base_channels * (2**(i-1)), bn=bn))
                setattr(self, f'decode{i}', Modules.Up3D(base_channels * (2**i), base_channels * (2**(i-1)), bn=bn))

        setattr(self, f'bottleneck{depth}',
                Modules.Down3D(base_channels * (2 ** (depth - 2)), base_channels * (2 ** (depth - 1)), bn=bn))

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        encode_features = []
        x = input

        for i in range(1, self.depth):
            x = getattr(self, f"encode{i}")(x)
            encode_features.append(x)

        x = getattr(self, f"bottleneck{self.depth}")(x)

        for i in reversed(range(1, self.depth)):
            x = getattr(self, f"decode{i}")(x, encode_features[i-1])

        output = self.out(x)
        return output