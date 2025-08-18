import torch
import torch.nn as nn
import torch.nn.init as I
import math
from .Modules.General_Components import ResBasicBlock, BasicBlock, sSE, FourierShells

# Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
# In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich,
# Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

# Çiçek Ö, Abdulkadir A, Lienkamp S S, et al. 3D U-Net: learning dense volumetric segmentation from sparse
# annotation[C]//Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International
# Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19. Springer International Publishing,
# 2016: 424-432.

# This file contains U-net-like Architectures for semantic segmentation,

# U-net: The most influential DL image classification paper.
# Introduced the concept of skip connection to feed spatial information into the decoder part of the network
# The original UNet implementation in the paper doesn't use batch normalization, nor does conv padding

class UNet(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, block_type='Basic', se=False, instance=False):
        super().__init__()
        self.depth = depth
        self.se = se
        self.instance = instance
        self.special_layers = round(math.log2(z_to_xy_ratio))

        # Validate parameters
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Z to XY ratio requires deeper network")
        if depth < 2:
            raise ValueError("Depth must be at least 2")

        # Precompute channel sizes
        channels = [min(base_channels * (2 ** i), 256) for i in range(depth)]

        # Precompute downsampling parameters
        kernel_sizes_conv = (3, 3, 3)
        scale_factors, kernel_sizes, paddings = [], [], []
        for i in range(depth - 1):
            if self.special_layers > 0 and i < self.special_layers:
                scale_factors.append((1, 2, 2))
                kernel_sizes.append((1, 4, 4))
                paddings.append((0, 1, 1))
            elif self.special_layers < 0 and i < -self.special_layers:
                scale_factors.append((2, 1, 1))
                kernel_sizes.append((4, 1, 1))
                paddings.append((1, 0, 0))
            else:
                scale_factors.append((2, 2, 2))
                kernel_sizes.append((4, 4, 4))
                paddings.append((1, 1, 1))

        # Precompute rescale factors for outputs
        rescale_factors = []
        for i in range(depth - 1):
            if self.special_layers > 0:
                z_factor = max(1, 2 ** (i - self.special_layers))
                xy_factor = 2 ** i
            else:
                z_factor = 2 ** i
                xy_factor = max(1, 2 ** (i + self.special_layers))
            rescale_factors.append((z_factor, xy_factor, xy_factor))

        # Select block type
        block_dict = {
            'Basic': BasicBlock,
            'Residual': ResBasicBlock,
        }
        block = block_dict[block_type]

        # Initialize module lists
        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.deconv_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.p_output_convs = nn.ModuleList()
        self.rescale_blocks = nn.ModuleList()
        self.encoder_se = nn.ModuleList() if se else [nn.Identity()] * (depth - 1)
        self.decoder_se = nn.ModuleList() if se else [nn.Identity()] * (depth - 1)

        # Build encoder path
        for i in range(depth - 1):
            if i == 0:
                encoder = nn.Sequential(nn.Conv3d(1, channels[i], kernel_sizes_conv, padding='same'),
                                        block(channels[i], channels[i], kernel_sizes_conv, num_conv=1))
            else:
                encoder = block(channels[i], channels[i], kernel_sizes_conv, num_conv=2)

            down = nn.Conv3d(
                channels[i], channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=scale_factors[i],
                padding=paddings[i]
            )

            self.encoder_blocks.append(encoder)
            self.down_blocks.append(down)

            if se:
                self.encoder_se.append(sSE(channels[i]))

        # Build bottleneck
        self.bottleneck = block(channels[-1], channels[-1], kernel_sizes_conv, num_conv=2)
        self.bottleneck_se = sSE(channels[-1]) if se else nn.Identity()

        # Build decoder path
        for i in range(depth - 1):
            deconv = nn.ConvTranspose3d(
                channels[i + 1], channels[i],
                kernel_size=kernel_sizes[i],
                stride=scale_factors[i],
                padding=paddings[i]
            )

            # Decoder blocks
            if i == 0:
                decoder = BasicBlock(channels[1], channels[0], kernel_sizes_conv, num_conv=2, norm=True)
                if self.instance:
                    self.c_output_conv = nn.Conv3d(channels[0], 1, kernel_size=1)
            else:
                decoder = block(channels[i], channels[i], kernel_sizes_conv, num_conv=2, norm=True)

            self.deconv_blocks.append(deconv)
            self.decoder_blocks.append(decoder)
            self.p_output_convs.append(nn.Conv3d(channels[i], 1, kernel_size=1))

            # Rescale blocks (only for i>0)
            if i > 0:
                self.rescale_blocks.append(
                    nn.Upsample(scale_factor=rescale_factors[i], mode='trilinear', align_corners=False)
                )
            else:
                self.rescale_blocks.append(nn.Identity())

            if se:
                self.decoder_se.append(sSE(channels[i]))

    def forward(self, x):
        # Encoder path
        encoder_features = []
        for i in range(self.depth - 1):
            x = self.encoder_blocks[i](x)
            x = self.encoder_se[i](x)
            encoder_features.append(x)
            x = self.down_blocks[i](x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_se(x)

        # Decoder path with outputs
        p_outputs = []
        for i in reversed(range(self.depth - 1)):
            x = self.deconv_blocks[i](x)

            # Combine with encoder features
            if i == 0:
                x = torch.cat([x, encoder_features[i]], dim=1)
            else:
                x = x + encoder_features[i]

            x = self.decoder_blocks[i](x)
            x = self.decoder_se[i](x)

            # Generate and rescale output
            out = self.p_output_convs[i](x)
            out = self.rescale_blocks[i](out)
            p_outputs.append(out)

        if self.instance:
            c_output = self.c_output_conv(x)
            return p_outputs, c_output
        else:
            return p_outputs


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # LeakyReLU everywhere → use Kaiming with a = 0.01 (default of LeakyReLU)
        I.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            I.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        # All the intermediate fcs feeding LeakyReLU
        I.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        I.zeros_(m.bias)