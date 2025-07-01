import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm
from .Modules.GANS_Components import (ConstantInput, PixelNorm, EqualizedLinear, StyledConv3d, ToG, ResBasicBlock, BasicBlock,
                                      LowPassFilter, MinibatchStd3d, SimpleGANConv, SimpleModulatedStyleConv, SimpleToG)


class Generator(nn.Module):
    def __init__(self, n_mlp=3, style_dim=128, final_resolution=(256, 256, 256), lr_mlp=0.01, base_channels=8, depth=5, z_to_xy_ratio=1, precision=torch.float32):
        super().__init__()
        torch.manual_seed(0)
        self.checkpointing = False
        self.style_dim = style_dim

        mapping_layers = [PixelNorm()]
        for i in range(n_mlp):
            mapping_layers.append(EqualizedLinear(style_dim, style_dim, lr_mul=lr_mlp, precision=precision))
            mapping_layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mapping = nn.Sequential(*mapping_layers)

        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        initial_resolution = (final_resolution[0]//2**(depth-self.special_layers-1),
                              final_resolution[1]//2**(depth-1),
                              final_resolution[2]//2**(depth-1))
        self.input = ConstantInput(min(base_channels*(2**(depth-1)), 256), initial_resolution, precision)

        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]
        kernel_sizes_conv_up = [(1, 4, 4) if self.special_layers > 0 and i < self.special_layers else
                                (4, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                (4, 4, 4) for i in range(depth)]

        for i in range(depth):
            multiplier_l = min(base_channels * (2 ** i), 256)
            multiplier_h = min(base_channels * (2 ** (i + 1)), 256)
            '''if self.special_layers > 0:
                depth_factor = max(1, 2 ** (i - self.special_layers))
                xy_factor = 2 ** i
            else:
                depth_factor = 2 ** i
                xy_factor = max(1, 2 ** (i - self.special_layers))'''
            if i != (depth-1):
                setattr(self, f'conv{i}_0', StyledConv3d(multiplier_h, multiplier_l, kernel_sizes_conv[i], style_dim, upsample=True, precision=precision))
                setattr(self, f'toG{i}', ToG(multiplier_l, style_dim, precision=precision))
            setattr(self, f'conv{i}_1', StyledConv3d(multiplier_l, multiplier_l, kernel_sizes_conv[i], style_dim, upsample=False, precision=precision))
        #self.norm = nn.BatchNorm3d(1)

    def forward(self, z, mixing_prob=0.9):
        batch_size = z.size(0)
        if self.training and torch.rand(1).item() < mixing_prob:
            # Style mixing
            z1 = z
            z2 = torch.randn_like(z1)
            w1 = self.mapping(z1)
            w2 = self.mapping(z2)
            cutoff = torch.randint(1, self.depth, (1,)).item()
        else:
            w1 = self.mapping(z)
            w2 = None
            cutoff = None
        x = self.input(batch_size)
        styles = []

        for i in reversed(range(self.depth)):
            if w2 is not None and i < cutoff:
                styles.append(w2)
            else:
                styles.append(w1)

        def run_initial_conv(x, style):
            x = getattr(self, f'conv{self.depth - 1}_1')(x, style[self.depth - 1])
            return x

        def run_other_conv(x, style, skip, i):
            x = getattr(self, f'conv{i}_0')(x, style[i])
            x = getattr(self, f'conv{i}_1')(x, style[i])
            skip = getattr(self, f'toG{i}')(x, style[i], skip)
            return x, skip

        if self.checkpointing and self.training:
            x = checkpoint(run_initial_conv, x, styles, use_reentrant=False)
        else:
            x = run_initial_conv(x, styles)
        skip = None
        for i in reversed(range(self.depth - 1)):
            if self.checkpointing and self.training:
                x, skip = checkpoint(run_other_conv, x, styles, skip, i, use_reentrant=False)
            else:
                x, skip = run_other_conv(x, styles, skip, i)

        return torch.tanh(skip) * 10

class GeneratorSimple(nn.Module):
    def __init__(self, n_mlp=3, final_resolution=(256, 256, 256), base_channels=8, depth=5, z_to_xy_ratio=1, precision=torch.float32):
        super().__init__()
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        self.initial_resolution = (final_resolution[0] // 2 ** (depth - self.special_layers - 1),
                                   final_resolution[1] // 2 ** (depth - 1),
                                   final_resolution[2] // 2 ** (depth - 1))
        self.max_channel = min((base_channels * (2 ** (depth - 1))), 512)
        self.mlp_dim = (self.initial_resolution[0] * self.initial_resolution[1] * self.initial_resolution[2])
        mlp_dim_final = self.mlp_dim * self.max_channel
        mlp_layers = []
        for i in range(n_mlp):
            mlp_layers.append(nn.Linear(self.mlp_dim, self.mlp_dim, bias=False))
            mlp_layers.append(nn.BatchNorm1d(self.mlp_dim))
            mlp_layers.append(nn.LeakyReLU(inplace=True))
        mlp_layers.append(nn.Linear(self.mlp_dim, mlp_dim_final, bias=False))
        mlp_layers.append(nn.Dropout(0.25))
        mlp_layers.append(nn.BatchNorm1d(mlp_dim_final))
        mlp_layers.append(nn.LeakyReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp_layers)
        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]
        kernel_sizes_conv_up = [(1, 4, 4) if self.special_layers > 0 and i < self.special_layers else
                                (4, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                (4, 4, 4) for i in range(depth)]
        for i in range(depth):
            multiplier_l = min(base_channels * (2 ** i), 512)
            multiplier_h = min(base_channels * (2 ** (i + 1)), 512)
            if i != (depth-1):
                setattr(self, f'conv{i}_0', SimpleGANConv(multiplier_h, multiplier_l, kernel_sizes_conv[i], upsample=True))
                #setattr(self, f'dropout{i}', nn.Dropout3d(0.25))
                setattr(self, f'toG{i}', SimpleToG(multiplier_l))
            setattr(self, f'conv{i}_1', SimpleGANConv(multiplier_l, multiplier_l, kernel_sizes_conv[i], upsample=False))
            setattr(self, f'conv{i}_2', SimpleGANConv(multiplier_l, multiplier_l, kernel_sizes_conv[i], upsample=False))

    def forward(self, z):
        w = self.mlp(z)
        w = w.reshape(z.size(0), self.max_channel, *self.initial_resolution)
        #print(w.mean(), w.std())
        x = getattr(self, f'conv{self.depth - 1}_1')(w)
        x = getattr(self, f'conv{self.depth - 1}_2')(x)
        #print(x.mean(), x.std())
        skip = None
        for i in reversed(range(self.depth - 1)):
            x = getattr(self, f'conv{i}_0')(x)
            #x = getattr(self, f'dropout{i}')(x)
            x = getattr(self, f'conv{i}_1')(x)
            x = getattr(self, f'conv{i}_2')(x)
        skip = getattr(self, f'toG{i}')(x, skip)
        return torch.tanh(skip)



class Discriminator(nn.Module):
    def __init__(self, resolution=(256, 256, 256), base_channels=8, depth=5, z_to_xy_ratio=1):
        super().__init__()
        torch.manual_seed(0)
        self.checkpointing = False
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
                setattr(self, f'conv{i}', BasicBlock(1, multiplier_l, kernel_sizes_conv[i], 2))
                setattr(self, f'down{i}', nn.Sequential(
                    (nn.Conv3d(multiplier_l, multiplier_h, scale_down_kernel_size[i], scale_down[i], padding_down[i], bias=False)),
                    #nn.Dropout3d(0.2),
                    nn.BatchNorm3d(multiplier_h),
                    nn.SELU(inplace=True)))
            else:
                setattr(self, f'conv{i}', ResBasicBlock(multiplier_l, multiplier_l, kernel_sizes_conv[i], 2))
                setattr(self, f'down{i}', nn.Sequential(
                    (nn.Conv3d(multiplier_l, multiplier_h, scale_down_kernel_size[i], scale_down[i], padding_down[i], bias=False)),
                    #nn.Dropout3d(0.2),
                    nn.BatchNorm3d(multiplier_h),
                    nn.SELU(inplace=True)))
        final_linear_dim = (multiplier_h + 1) * smallest_resolution[0]*smallest_resolution[1]*smallest_resolution[2]
        self.final_linear = nn.Sequential(
            nn.Dropout3d(0.25),
            MinibatchStd3d(),
            nn.Flatten(),
            (nn.Linear(final_linear_dim, multiplier_h)),
            nn.BatchNorm1d(multiplier_h),
            nn.SELU(inplace=True),
            (nn.Linear(multiplier_h, multiplier_h)),
            nn.BatchNorm1d(multiplier_h),
            nn.SELU(inplace=True),
            nn.Linear(multiplier_h, 1)
        )

    def forward(self, x):
        def run_conv(x, i):
            x = getattr(self, f'conv{i}')(x)
            x = getattr(self, f'down{i}')(x)
            return x

        for i in range(self.depth):
            if self.checkpointing == True and self.training:
                x = checkpoint(run_conv, x, i, use_reentrant=False)
            else:
                x = run_conv(x, i)

        x = self.final_linear(x)
        return x