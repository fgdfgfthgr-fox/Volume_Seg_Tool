import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules.GANS_Components import ConstantInput, PixelNorm, EqualizedLinear, ModulatedConv3d, NoiseInjection, StyledConv3d, ToG, MinibatchStd3d
from .Modules.General_Components import ResBasicBlock, BasicBlock


class Generator(nn.Module):
    def __init__(self, n_mlp=3, style_dim=128, final_resolution=(256, 256, 256), lr_mlp=0.01, base_channels=8, depth=5, z_to_xy_ratio=1):
        super().__init__()
        self.style_dim = style_dim

        mapping_layers = [PixelNorm()]
        for i in range(n_mlp):
            mapping_layers.append(EqualizedLinear(style_dim, style_dim, lr_mul=lr_mlp))
            mapping_layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mapping = nn.Sequential(*mapping_layers)

        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        initial_resolution = (final_resolution[0]//2**(depth-self.special_layers-1),
                              final_resolution[1]//2**(depth-1),
                              final_resolution[2]//2**(depth-1))
        self.input = ConstantInput(min(base_channels*(2**depth), 256), initial_resolution)

        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]

        for i in range(depth):
            multiplier_l = min(base_channels * (2 ** i), 256)
            multiplier_h = min(base_channels * (2 ** (i + 1)), 256)
            upsample = False if i == (depth-1) else True
            if self.special_layers > 0:
                depth_factor = max(1, 2 ** (i - self.special_layers))
                xy_factor = 2 ** i
            else:
                depth_factor = 2 ** i
                xy_factor = max(1, 2 ** (i - self.special_layers))
            setattr(self, f'conv{i}', StyledConv3d(multiplier_h, multiplier_l, kernel_sizes_conv[i], style_dim, upsample=upsample))
            setattr(self, f'toG{i}', ToG(multiplier_l, style_dim))
            setattr(self, f'rescale{i}',
                    nn.Upsample(scale_factor=(depth_factor, xy_factor, xy_factor), mode='trilinear', align_corners=False))

    def forward(self, batch_size):
        z = torch.randn((batch_size, self.style_dim))
        w = self.mapping(z)

        x = self.input(batch_size)
        outputs = []
        for i in reversed(range(self.depth)):
            x = getattr(self, f'conv{i}')(x, w)
            out = getattr(self, f'toG{i}')(x, w)
            if i != 0:
                out = getattr(self, f'rescale{i}')(out)
            outputs.append(out)
        outputs = torch.sum(torch.stack(outputs), dim=0)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, resolution=(256, 256, 256), base_channels=8, depth=5, z_to_xy_ratio=1):
        super().__init__()
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))

        smallest_resolution = (resolution[0]//2**(depth-self.special_layers),
                               resolution[1]//2**(depth),
                               resolution[2]//2**(depth))
        scale_down = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                      (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                      (2, 2, 2) for i in range(depth)]
        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]
        scale_down_kernel_size = [(1, 4, 4) if self.special_layers > 0 and i < self.special_layers else
                                  (4, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                  (4, 4, 4) for i in range(depth)]
        padding_down = [(0, 1, 1) if self.special_layers > 0 and i < self.special_layers else
                        (1, 0, 0) if self.special_layers < 0 and i < -self.special_layers else
                        (1, 1, 1) for i in range(depth)]

        for i in range(depth):
            multiplier_l = min(base_channels * (2 ** i), 256)
            multiplier_h = min(base_channels * (2 ** (i + 1)), 256)
            if i == 0:
                setattr(self, f'conv{i}', BasicBlock(1, multiplier_l, kernel_sizes_conv[i]))
            else:
                setattr(self, f'conv{i}', ResBasicBlock(multiplier_l, multiplier_l, kernel_sizes_conv[i]))

            setattr(self, f'down{i}', nn.Sequential(
                nn.Conv3d(multiplier_l, multiplier_h, scale_down_kernel_size[i], scale_down[i], padding_down[i]),
                nn.InstanceNorm3d(multiplier_h),
                nn.SiLU(inplace=True)))

        self.final_linear = nn.Sequential(
            MinibatchStd3d(),
            nn.Flatten(),
            EqualizedLinear((multiplier_h + 1) * smallest_resolution[0] * smallest_resolution[1] * smallest_resolution[2], multiplier_h),
            nn.LeakyReLU(0.2),
            EqualizedLinear(multiplier_h, 1),
        )

    def forward(self, x):
        for i in range(self.depth):
            x = getattr(self, f'conv{i}')(x)
            x = getattr(self, f'down{i}')(x)
        x = self.final_linear(x)
        return x