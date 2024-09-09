import torch
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
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic',
                 se=False, unsupervised=False, label_mean=torch.tensor(0.5)):
        super(UNet, self).__init__()
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        self.se = se
        weights = torch.linspace(0.5, 1.0, self.depth)
        weights /= weights.sum()
        self.d_s_weights = weights

        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio requires a deeper network.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")

        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))

        kernel_sizes_conv = (3, 3, 3)

        kernel_sizes_transpose = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                                  (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                  (2, 2, 2) for i in range(depth)]
        block = {'Basic': BasicBlock, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        block_top = {'Basic': BasicBlockSingle, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        block_top(1, base_channels, kernel_sizes_conv))
                if se: setattr(self, f'encode_se0', scSE(base_channels))
                setattr(self, f'decode0',
                        block_top(base_channels, base_channels, kernel_sizes_conv))
                if se: setattr(self, f'decode_se0', scSE(base_channels))
                setattr(self, f'up0',
                        nn.ConvTranspose3d(2 * base_channels, base_channels,
                                           kernel_sizes_transpose[0], kernel_sizes_transpose[0]))
                if unsupervised:
                    setattr(self, f'u_decode0',
                            block_top(base_channels, base_channels, kernel_sizes_conv))
                    if se: setattr(self, f'u_decode_se0', scSE(base_channels))
                    setattr(self, f'u_up0',
                            nn.ConvTranspose3d(2 * base_channels, base_channels,
                                               kernel_sizes_transpose[0], kernel_sizes_transpose[0]))
            elif i == depth-1:
                setattr(self, 'bottleneck',
                        block(base_channels * (2 ** (i - 1)), (base_channels * (2 ** i)), kernel_sizes_conv))
                if se: setattr(self, f'bottleneck_se', scSE(base_channels * (2 ** i)))
            else:
                setattr(self, f'encode{i}',
                        block(base_channels * (2 ** (i-1)), (base_channels * (2 ** i)), kernel_sizes_conv))
                if se: setattr(self, f'encode_se{i}', scSE(base_channels * (2 ** i)))
                setattr(self, f'decode{i}',
                        block(base_channels * (2 ** i), (base_channels * (2 ** i)), kernel_sizes_conv))
                if se: setattr(self, f'decode_se{i}', scSE(base_channels * (2 ** i)))
                setattr(self, f'up{i}',
                        nn.ConvTranspose3d((base_channels * (2 ** (i + 1))), (base_channels * (2 ** i)),
                                           kernel_sizes_transpose[i], kernel_sizes_transpose[i]))
                if unsupervised:
                    setattr(self, f'u_decode{i}',
                            block(base_channels * (2 ** i), (base_channels * (2 ** i)), kernel_sizes_conv))
                    if se: setattr(self, f'u_decode_se{i}', scSE(base_channels * (2 ** i)))
                    setattr(self, f'u_up{i}',
                            nn.ConvTranspose3d((base_channels * (2 ** (i + 1))), (base_channels * (2 ** i)),
                                               kernel_sizes_transpose[i], kernel_sizes_transpose[i]))

        logit_label_mean = torch.log(label_mean / (1 - label_mean)) * 0.5
        self.s_out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                   nn.Sigmoid())
        with torch.no_grad():
            self.s_out[0].bias.fill_(logit_label_mean)
        if unsupervised:
            self.u_out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                       nn.Sigmoid())

    def semantic_decode(self, bottleneck, encode_features):
        s_x = bottleneck

        for i in reversed(range(self.depth - 1)):

            s_x = getattr(self, f"up{i}")(s_x)
            if i > 0:  # Skip connection for all but the first layer
                s_x += encode_features[i - 1]
                s_x = getattr(self, f"decode{i}")(s_x)
            else:
                s_x = getattr(self, f"decode{i}")(s_x)

            if self.se:
                s_x = getattr(self, f"decode_se{i}")(s_x)

        output = self.s_out(s_x)
        return output

    def unsupervised_decode(self, bottleneck):
        for i in reversed(range(self.depth - 1)):
            if i == self.depth - 2:
                u_x = getattr(self, f"u_up{i}")(bottleneck)
            else:
                u_x = getattr(self, f"u_up{i}")(u_x)
            u_x = getattr(self, f"u_decode{i}")(u_x)
            if self.se:
                u_x = getattr(self, f"u_decode_se{i}")(u_x)
        u_output = self.u_out(u_x)
        return u_output

    def forward(self, input, type=(1,)):
        encode_features = []
        x = input

        for i in range(self.depth - 1):
            x = getattr(self, f"encode{i}")(x)
            if self.se: x = getattr(self, f"encode_se{i}")(x)
            if i != 0:
                encode_features.append(x)
            x = self.max_pool_flat(x) if self.special_layers > 0 and i < self.special_layers else \
                self.max_pool_shrink(x) if self.special_layers < 0 and i < -self.special_layers else \
                self.max_pool(x)

        bottleneck = getattr(self, "bottleneck")(x)
        if self.se: bottleneck = getattr(self, f"bottleneck_se")(bottleneck)

        if type[0] == 0:
            return self.semantic_decode(bottleneck, encode_features)
        elif type[0] == 1:
            return self.unsupervised_decode(bottleneck)
        elif type[0] == 2:
            return [self.semantic_decode(bottleneck, encode_features), self.unsupervised_decode(bottleneck)]
        else:
            raise ValueError("Invalid data type. Should be either '0'(normal) or '1'(unsupervised) or '2'(both).")
