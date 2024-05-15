import torch.nn as nn
import math
from .Modules.General_Components import ResBasicBlock, ResBottleneckBlock, BasicBlock, BasicBlockSingle, scSE

# Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
# In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich,
# Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

# Çiçek Ö, Abdulkadir A, Lienkamp S S, et al. 3D U-Net: learning dense volumetric segmentation from sparse
# annotation[C]//Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International
# Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19. Springer International Publishing,
# 2016: 424-432.

# This file contains U-net-like Architectures for semantic segmentation,
# with the main difference being the skip connection adds the channels instead of concatenate them
# Basic: Use normal convolution block
# Residual: Use residual block
# ResidualBottleneck: Use residual bottleneck block

# U-net: The most influential DL image classification paper.
# Introduced the concept of skip connection to feed spatial information into the decoder part of the network
# The original UNet implementation in the paper doesn't use batch normalization, nor does conv padding


class UNet(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic', se=False):
        super(UNet, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        self.se = se
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = [(3, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 3, 3) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]

        kernel_sizes_transpose = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                                  (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                  (2, 2, 2) for i in range(depth)]
        block = {'Basic': BasicBlock, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        block_top = {'Basic': BasicBlockSingle, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        block_top(1, base_channels, kernel_sizes_conv[0]))
                if se: setattr(self, f'encode_se0', scSE(base_channels))
                setattr(self, f'decode0',
                        block_top(base_channels, base_channels, kernel_sizes_conv[0]))
                if se: setattr(self, f'decode_se0', scSE(base_channels))
                setattr(self, f'up0',
                        nn.ConvTranspose3d(2*base_channels, base_channels, kernel_sizes_transpose[0], kernel_sizes_transpose[0]))
            elif i == depth-1:
                setattr(self, 'bottleneck',
                        block(base_channels * (2 ** (i - 1)), (base_channels * (2 ** i)),
                              kernel_sizes_conv[i]))
                if se: setattr(self, f'bottleneck_se', scSE(base_channels * (2 ** i)))
            else:
                setattr(self, f'encode{i}',
                        block(base_channels * (2 ** (i-1)), (base_channels * (2 ** i)), kernel_sizes_conv[i]))
                if se: setattr(self, f'encode_se{i}', scSE(base_channels * (2 ** i)))
                setattr(self, f'decode{i}',
                        block(base_channels * (2 ** i), (base_channels * (2 ** i)), kernel_sizes_conv[i]))
                if se: setattr(self, f'decode_se{i}', scSE(base_channels * (2 ** i)))
                setattr(self, f'up{i}',
                        nn.ConvTranspose3d((base_channels * (2 ** (i + 1))), (base_channels * (2 ** i)),
                                           kernel_sizes_transpose[i], kernel_sizes_transpose[i]))

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        encode_features = []
        x = input

        for i in range(self.depth - 1):
            x = getattr(self, f"encode{i}")(x)
            if self.se: x = getattr(self, f"encode_se{i}")(x)
            encode_features.append(x)
            x = self.max_pool_flat(x) if self.special_layers > 0 and i < self.special_layers else \
                self.max_pool_shrink(x) if self.special_layers < 0 and i < -self.special_layers else \
                self.max_pool(x)

        bottleneck = getattr(self, "bottleneck")(x)
        if self.se: bottleneck = getattr(self, f"bottleneck_se")(bottleneck)

        for i in reversed(range(self.depth - 1)):
            if i == self.depth-2:
                x = getattr(self, f"up{i}")(bottleneck)
            else:
                x = getattr(self, f"up{i}")(x)
            x = x + encode_features[i]
            x = getattr(self, f"decode{i}")(x)
            if self.se: x = getattr(self, f"decode_se{i}")(x)

        output = self.out(x)

        return output
