import torch.nn as nn
import math
from .Modules.General_Components import ResBasicBlock, ResBottleneckBlock, BasicBlock

# Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
# In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich,
# Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

# Çiçek Ö, Abdulkadir A, Lienkamp S S, et al. 3D U-Net: learning dense volumetric segmentation from sparse
# annotation[C]//Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International
# Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19. Springer International Publishing,
# 2016: 424-432.

# This file contains U-net-like Architectures for instance segmentation,
# with the main difference being the skip connection adds the channels instead of concatenate them
# Two branches, p(ixel) branch predict the mask, which c(ontour) branch predict contour of objects
# The two branches shares the same encoder
# Basic: Use normal convolution block
# Residual: Use residual block
# ResidualBottleneck: Use residual bottleneck block

# U-net: The most influential DL image classification paper.
# Introduced the concept of skip connection to feed spatial information into the decoder part of the network
# The original UNet implementation in the paper doesn't use batch normalization, nor does conv padding


class Basic(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1):
        super(Basic, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = []
        kernel_sizes_transpose = []
        for i in range(depth):
            if self.special_layers > 0 and i < self.special_layers:
                kernel_sizes_conv.append((1, 3, 3))
                kernel_sizes_transpose.append((1, 2, 2))
            elif self.special_layers < 0 and i < -self.special_layers:
                kernel_sizes_conv.append((3, 1, 1))
                kernel_sizes_transpose.append((2, 1, 1))
            else:
                kernel_sizes_conv.append((3, 3, 3))
                kernel_sizes_transpose.append((2, 2, 2))
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        BasicBlock(1, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'p_decode0',
                        BasicBlock(base_channels, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'c_decode0',
                        BasicBlock(base_channels, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'p_up0',
                        nn.LazyConvTranspose3d(base_channels, kernel_sizes_transpose[0], kernel_sizes_transpose[0]))
                setattr(self, f'c_up0',
                        nn.LazyConvTranspose3d(base_channels, kernel_sizes_transpose[0], kernel_sizes_transpose[0]))
            elif i == depth-1:
                setattr(self, 'bottleneck',
                        BasicBlock(base_channels * (2 ** (depth-2)), (base_channels * (2 ** (depth-1))),
                                   kernel_sizes_conv[i]))
            else:
                setattr(self, f'encode{i}',
                        BasicBlock(base_channels * (2 ** (i-1)), (base_channels * (2 ** i)), kernel_sizes_conv[i]))
                setattr(self, f'p_decode{i}',
                        BasicBlock(base_channels * (2 ** i), (base_channels * (2 ** (i-1))), kernel_sizes_conv[i]))
                setattr(self, f'c_decode{i}',
                        BasicBlock(base_channels * (2 ** i), (base_channels * (2 ** (i-1))), kernel_sizes_conv[i]))
                setattr(self, f'p_up{i}',
                        nn.LazyConvTranspose3d((base_channels * (2 ** i)),
                                               kernel_sizes_transpose[i], kernel_sizes_transpose[i]))
                setattr(self, f'c_up{i}',
                        nn.LazyConvTranspose3d((base_channels * (2 ** i)),
                                               kernel_sizes_transpose[i], kernel_sizes_transpose[i]))

        self.p_out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.c_out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        encode_features = []
        x = input

        for i in range(self.depth - 1):
            x = getattr(self, f"encode{i}")(x)
            encode_features.append(x)
            if self.special_layers > 0 and i < self.special_layers:
                x = self.max_pool_flat(x)
            elif self.special_layers < 0 and i < -self.special_layers:
                x = self.max_pool_shrink(x)
            else:
                x = self.max_pool(x)

        bottleneck = getattr(self, "bottleneck")(x)

        for i in reversed(range(self.depth - 1)):
            if i == self.depth-2:
                p_x = getattr(self, f"p_up{i}")(bottleneck)
                c_x = getattr(self, f"c_up{i}")(bottleneck)
            else:
                p_x = getattr(self, f"p_up{i}")(p_x)
                c_x = getattr(self, f"c_up{i}")(c_x)
            p_x = p_x + encode_features[i]
            c_x = c_x + encode_features[i]
            p_x = getattr(self, f"p_decode{i}")(p_x)
            c_x = getattr(self, f"c_decode{i}")(c_x)

        p_output = self.p_out(p_x)
        c_output = self.c_out(c_x)

        return p_output, c_output


class Residual(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1):
        super(Residual, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = []
        kernel_sizes_transpose = []
        for i in range(depth):
            if self.special_layers > 0 and i < self.special_layers:
                kernel_sizes_conv.append((1, 3, 3))
                kernel_sizes_transpose.append((1, 2, 2))
            elif self.special_layers < 0 and i < -self.special_layers:
                kernel_sizes_conv.append((3, 1, 1))
                kernel_sizes_transpose.append((2, 1, 1))
            else:
                kernel_sizes_conv.append((3, 3, 3))
                kernel_sizes_transpose.append((2, 2, 2))
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        ResBasicBlock(1, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'p_decode0',
                        ResBasicBlock(base_channels, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'c_decode0',
                        ResBasicBlock(base_channels, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'p_up0',
                        nn.LazyConvTranspose3d(base_channels, kernel_sizes_transpose[0], kernel_sizes_transpose[0]))
                setattr(self, f'c_up0',
                        nn.LazyConvTranspose3d(base_channels, kernel_sizes_transpose[0], kernel_sizes_transpose[0]))
            elif i == depth-1:
                setattr(self, 'bottleneck',
                        ResBasicBlock(base_channels * (2 ** (depth-2)), (base_channels * (2 ** (depth-1))),
                                      kernel_sizes_conv[i]))
            else:
                setattr(self, f'encode{i}',
                        ResBasicBlock(base_channels * (2 ** (i-1)), (base_channels * (2 ** i)), kernel_sizes_conv[i]))
                setattr(self, f'p_decode{i}',
                        ResBasicBlock(base_channels * (2 ** i), (base_channels * (2 ** (i-1))), kernel_sizes_conv[i]))
                setattr(self, f'c_decode{i}',
                        ResBasicBlock(base_channels * (2 ** i), (base_channels * (2 ** (i-1))), kernel_sizes_conv[i]))
                setattr(self, f'p_up{i}',
                        nn.LazyConvTranspose3d((base_channels * (2 ** i)),
                                               kernel_sizes_transpose[i], kernel_sizes_transpose[i]))
                setattr(self, f'c_up{i}',
                        nn.LazyConvTranspose3d((base_channels * (2 ** i)),
                                               kernel_sizes_transpose[i], kernel_sizes_transpose[i]))

        self.p_out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.c_out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        encode_features = []
        x = input

        for i in range(self.depth - 1):
            x = getattr(self, f"encode{i}")(x)
            encode_features.append(x)
            if self.special_layers > 0 and i < self.special_layers:
                x = self.max_pool_flat(x)
            elif self.special_layers < 0 and i < -self.special_layers:
                x = self.max_pool_shrink(x)
            else:
                x = self.max_pool(x)

        bottleneck = getattr(self, "bottleneck")(x)

        for i in reversed(range(self.depth - 1)):
            if i == self.depth-2:
                p_x = getattr(self, f"p_up{i}")(bottleneck)
                c_x = getattr(self, f"c_up{i}")(bottleneck)
            else:
                p_x = getattr(self, f"p_up{i}")(p_x)
                c_x = getattr(self, f"c_up{i}")(c_x)
            p_x = p_x + encode_features[i]
            c_x = c_x + encode_features[i]
            p_x = getattr(self, f"p_decode{i}")(p_x)
            c_x = getattr(self, f"c_decode{i}")(c_x)

        p_output = self.p_out(p_x)
        c_output = self.c_out(c_x)

        return p_output, c_output


class ResidualBottleneck(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1):
        super(ResidualBottleneck, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = []
        kernel_sizes_transpose = []
        for i in range(depth):
            if self.special_layers > 0 and i < self.special_layers:
                kernel_sizes_conv.append((1, 3, 3))
                kernel_sizes_transpose.append((1, 2, 2))
            elif self.special_layers < 0 and i < -self.special_layers:
                kernel_sizes_conv.append((3, 1, 1))
                kernel_sizes_transpose.append((2, 1, 1))
            else:
                kernel_sizes_conv.append((3, 3, 3))
                kernel_sizes_transpose.append((2, 2, 2))
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        ResBottleneckBlock(1, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'p_decode0',
                        ResBottleneckBlock(base_channels, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'c_decode0',
                        ResBottleneckBlock(base_channels, base_channels, kernel_sizes_conv[0]))
                setattr(self, f'p_up0',
                        nn.LazyConvTranspose3d(base_channels, kernel_sizes_transpose[0], kernel_sizes_transpose[0]))
                setattr(self, f'c_up0',
                        nn.LazyConvTranspose3d(base_channels, kernel_sizes_transpose[0], kernel_sizes_transpose[0]))
            elif i == depth-1:
                setattr(self, 'bottleneck',
                        ResBottleneckBlock(base_channels * (2 ** (depth-2)), (base_channels * (2 ** (depth-1))),
                                           kernel_sizes_conv[i]))
            else:
                setattr(self, f'encode{i}',
                        ResBottleneckBlock(base_channels * (2 ** (i-1)), (base_channels * (2 ** i)), kernel_sizes_conv[i]))
                setattr(self, f'p_decode{i}',
                        ResBottleneckBlock(base_channels * (2 ** i), (base_channels * (2 ** (i-1))), kernel_sizes_conv[i]))
                setattr(self, f'c_decode{i}',
                        ResBottleneckBlock(base_channels * (2 ** i), (base_channels * (2 ** (i-1))), kernel_sizes_conv[i]))
                setattr(self, f'p_up{i}',
                        nn.LazyConvTranspose3d((base_channels * (2 ** i)),
                                               kernel_sizes_transpose[i], kernel_sizes_transpose[i]))
                setattr(self, f'c_up{i}',
                        nn.LazyConvTranspose3d((base_channels * (2 ** i)),
                                               kernel_sizes_transpose[i], kernel_sizes_transpose[i]))

        self.p_out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.c_out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        encode_features = []
        x = input

        for i in range(self.depth - 1):
            x = getattr(self, f"encode{i}")(x)
            encode_features.append(x)
            if self.special_layers > 0 and i < self.special_layers:
                x = self.max_pool_flat(x)
            elif self.special_layers < 0 and i < -self.special_layers:
                x = self.max_pool_shrink(x)
            else:
                x = self.max_pool(x)

        bottleneck = getattr(self, "bottleneck")(x)

        for i in reversed(range(self.depth - 1)):
            if i == self.depth-2:
                p_x = getattr(self, f"p_up{i}")(bottleneck)
                c_x = getattr(self, f"c_up{i}")(bottleneck)
            else:
                p_x = getattr(self, f"p_up{i}")(p_x)
                c_x = getattr(self, f"c_up{i}")(c_x)
            p_x = p_x + encode_features[i]
            c_x = c_x + encode_features[i]
            p_x = getattr(self, f"p_decode{i}")(p_x)
            c_x = getattr(self, f"c_decode{i}")(c_x)

        p_output = self.p_out(p_x)
        c_output = self.c_out(c_x)

        return p_output, c_output
