import torch
import torch.nn as nn


class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), num_conv=2):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        layers = []
        if in_channels == out_channels:
            self.mapping = nn.Identity()
        else:
            self.mapping = nn.Conv3d(in_channels, out_channels, 1, padding)
        for i in range(num_conv):
            layers.append(nn.Conv3d(in_channels if i == 0 else out_channels, out_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.InstanceNorm3d(out_channels))
            if i != num_conv - 1:
                layers.append(nn.SiLU(inplace=True))
        self.operations = nn.Sequential(*layers)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.mapping(x) + self.operations(x))


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), neck=4, num_conv=1):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        middle_channel = out_channels//neck
        self.conv_down = nn.Conv3d(in_channels, middle_channel, kernel_size=1, bias=False)

        layers = []
        for i in range(num_conv):
            layers.append(
                nn.Conv3d(middle_channel, middle_channel, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.InstanceNorm3d(middle_channel))
            if i != num_conv - 1:
                layers.append(nn.SiLU(inplace=True))
        self.convs = nn.Sequential(*layers)

        self.conv_up = nn.Conv3d(middle_channel, out_channels, kernel_size=1)
        self.bn = nn.InstanceNorm3d(middle_channel)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        b_x = self.silu(self.bn(self.conv_down(x)))
        b_x = self.convs(b_x)
        b_x = self.conv_up(b_x)
        return x + b_x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), num_conv=2, padding_mode='zeros'):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        layers = []
        for i in range(num_conv):
            layers.append(nn.Conv3d(in_channels if i == 0 else out_channels, out_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False, padding_mode=padding_mode))
            layers.append(nn.InstanceNorm3d(out_channels))
            layers.append(nn.SiLU(inplace=True))
        self.operations = nn.Sequential(*layers)

    def forward(self, x):
        return self.operations(x)


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.silu = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 2, in_channels)

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        x_avg = self.avg_pool(x).view(batch_size, num_channels)
        x_avg = self.silu(self.fc1(x_avg))
        x_avg = torch.sigmoid(self.fc2(x_avg))
        return x * (x_avg.view(batch_size, num_channels, 1, 1, 1))


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        x_spatial = torch.sigmoid(self.fc(x))
        return x * x_spatial


# Concurrent Spatial and Channel Squeeze & Excitation
# https://arxiv.org/abs/1803.02579
class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cse = cSE(in_channels)
        self.sse = sSE(in_channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


# Attention Block from Attention UNet
# https://arxiv.org/abs/1804.03999
class AttentionBlock(nn.Module):
    def __init__(self, g_channel, f_channel):
        super().__init__()
        intermediate_channel = g_channel // 2
        self.w_g = nn.Sequential(
            nn.Conv3d(g_channel, intermediate_channel, kernel_size=1, bias=False),
            nn.InstanceNorm3d(intermediate_channel)
        )

        self.w_x = nn.Sequential(
            nn.Conv3d(f_channel, intermediate_channel, kernel_size=1, bias=False),
            nn.InstanceNorm3d(intermediate_channel)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(intermediate_channel, 1, kernel_size=1, bias=False),
            nn.InstanceNorm3d(1),
            nn.Sigmoid(),
        )

        self.silu = nn.SiLU(inplace=True)

    def forward(self, gate, x):
        g1 = self.w_g(gate)
        x1 = self.w_x(x)
        return self.psi(self.silu(g1 + x1)) * x


# https://www-sciencedirect-com.ezproxy.otago.ac.nz/science/article/pii/S095741742401594X
class FeatureFusionModule(nn.Module):
    def __init__(self, num_features, ini_channels):
        super().__init__()
        feature_channels = [ini_channels * (2 ** i) for i in range(num_features)]
        for i in range(num_features):
            for j in range(num_features):
                if j > i:
                    # Downsamplings
                    difference = 2 ** (j - i)
                    setattr(self, f'{i}_to_{j}_down',
                            nn.Sequential(nn.Conv3d(feature_channels[i], feature_channels[i], difference, difference, bias=False),
                                          nn.InstanceNorm3d(feature_channels[i]),
                                          nn.Conv3d(feature_channels[i], feature_channels[j], 1, bias=False),
                                          nn.InstanceNorm3d(feature_channels[j])))
                elif i > j:
                    # Upsamplings
                    difference = 2 ** (i - j)
                    setattr(self, f'{i}_to_{j}_up',
                            nn.Sequential(nn.Upsample(scale_factor=difference, mode='trilinear', align_corners=True),
                                          nn.Conv3d(feature_channels[i], feature_channels[j], 1, bias=False),
                                          nn.InstanceNorm3d(feature_channels[j])))
            setattr(self, f'bn_{i}', nn.InstanceNorm3d(feature_channels[i]))
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature_maps):
        # feature_maps is a list of features from different resolution branches
        num_features = len(feature_maps)
        fused_features = [f.clone() for f in feature_maps]  # Start with the original feature maps

        for i in range(num_features):
            for j in range(num_features):
                if i > j:
                    # Upsample i to match the resolution of j
                    upsample = getattr(self, f'{i}_to_{j}_up')
                    fused_features[j] = fused_features[j] + upsample(feature_maps[i])
                elif j > i:
                    # Downsample i to match the resolution of j
                    downsample = getattr(self, f'{i}_to_{j}_down')
                    fused_features[j] = fused_features[j] + downsample(feature_maps[i])
        for i in range(num_features):
            fused_features[i] = getattr(self, f'bn_{i}')(fused_features[i])

        # Apply activation after fusion
        fused_features = [self.activation(f) for f in fused_features]

        return fused_features