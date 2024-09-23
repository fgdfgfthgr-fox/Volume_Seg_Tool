import torch.nn as nn
import math
from .Modules.General_Components import ResBasicBlock, ResBottleneckBlock, BasicBlock, scSE, sSE

# Models for various testing purpose


class Tiniest(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1):
        super(Tiniest, self).__init__()
        self.model = BasicBlock(1, base_channels)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        x = self.model(input)

        output = self.out(x)

        return output

class MaxUnpool(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic', se=False):
        super(MaxUnpool, self).__init__()
        self.max_pool = nn.MaxPool3d(2, ceil_mode=True, return_indices=True)
        self.max_unpool = nn.MaxUnpool3d(2)
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        self.se = se
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]
        block = {'Basic': BasicBlock, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        block(1, base_channels, kernel_sizes_conv[0]))
                if se: setattr(self, f'encode_se0', scSE(base_channels))
                setattr(self, f'decode0',
                        block(base_channels, base_channels, kernel_sizes_conv[0]))
                if se: setattr(self, f'decode_se0', scSE(base_channels))
            elif i == depth-1:
                setattr(self, 'bottleneck',
                        block(base_channels * (2 ** (i - 1)), (base_channels * (2 ** (i - 1))),
                              kernel_sizes_conv[i]))
                if se: setattr(self, f'bottleneck_se', scSE(base_channels * (2 ** (i-1))))
            else:
                setattr(self, f'encode{i}',
                        block(base_channels * (2 ** (i-1)), (base_channels * (2 ** i)), kernel_sizes_conv[i]))
                if se: setattr(self, f'encode_se{i}', scSE(base_channels * (2 ** i)))
                setattr(self, f'decode{i}',
                        block(base_channels * (2 ** i), (base_channels * (2 ** (i-1))), kernel_sizes_conv[i]))
                if se: setattr(self, f'decode_se{i}', scSE(base_channels * (2 ** (i-1))))

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, input):
        encode_features = []
        encode_indices = []
        x = input

        for i in range(self.depth - 1):
            x = getattr(self, f"encode{i}")(x)
            if self.se: x = getattr(self, f"encode_se{i}")(x)
            encode_features.append(x)
            x, indices = self.max_pool(x)
            encode_indices.append(indices)

        bottleneck = getattr(self, "bottleneck")(x)
        if self.se: bottleneck = getattr(self, f"bottleneck_se")(bottleneck)

        for i in reversed(range(self.depth - 1)):
            if i == self.depth-2:
                x = self.max_unpool(bottleneck, encode_indices[i])
            else:
                x = self.max_unpool(x, encode_indices[i])
            x = x + encode_features[i]
            x = getattr(self, f"decode{i}")(x)
            if self.se: x = getattr(self, f"decode_se{i}")(x)

        output = self.out(x)

        return output

class EncoderConvOnly(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic', se=False):
        super(EncoderConvOnly, self).__init__()
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
        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]

        kernel_sizes_transpose = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                                  (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                  (2, 2, 2) for i in range(depth)]
        linear_kernel = (1, 1, 1)
        block = {'Basic': BasicBlock, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        block(1, base_channels, kernel_sizes_conv[0]))
                if se: setattr(self, f'encode_se0', scSE(base_channels))
                setattr(self, f'decode0',
                        block(base_channels, base_channels, linear_kernel))
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
                        block(base_channels * (2 ** i), (base_channels * (2 ** i)), linear_kernel))
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


class LastLayerLinear(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic', se=False):
        super(LastLayerLinear, self).__init__()
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
        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]

        kernel_sizes_transpose = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                                  (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                  (2, 2, 2) for i in range(depth)]
        linear_kernel = (1, 1, 1)
        block = {'Basic': BasicBlock, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        block(1, base_channels, kernel_sizes_conv[0]))
                if se: setattr(self, f'encode_se0', scSE(base_channels))
                setattr(self, f'decode0',
                        block(base_channels, base_channels, linear_kernel))
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


class LastLayerLinearSimplified(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic', se=False):
        super(LastLayerLinearSimplified, self).__init__()
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
        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]

        kernel_sizes_transpose = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                                  (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                  (2, 2, 2) for i in range(depth)]
        linear_kernel = (1, 1, 1)
        block = {'Basic': BasicBlock, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        block(1, base_channels, kernel_sizes_conv[0]))
                if se: setattr(self, f'encode_se0', scSE(base_channels))
                setattr(self, f'decode0',
                        nn.Conv3d(base_channels, 1, kernel_size=1))
                if se: setattr(self, f'decode_se0', sSE(1))
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

        self.out = nn.Sigmoid()

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


class SingleTopLayer(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic', se=False):
        super(SingleTopLayer, self).__init__()
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
        kernel_sizes_conv = [(1, 3, 3) if self.special_layers > 0 and i < self.special_layers else
                             (3, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (3, 3, 3) for i in range(depth)]

        kernel_sizes_transpose = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                                  (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                  (2, 2, 2) for i in range(depth)]
        block = {'Basic': BasicBlock, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        BasicBlockSingle(1, base_channels, kernel_sizes_conv[0]))
                if se: setattr(self, f'encode_se0', scSE(base_channels))
                setattr(self, f'decode0',
                        BasicBlockSingle(base_channels, base_channels, kernel_sizes_conv[0]))
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
