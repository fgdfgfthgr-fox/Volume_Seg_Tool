import torch.nn as nn
import math


class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros', bias=False)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros')
        self.bn = nn.InstanceNorm3d(out_channels, track_running_stats=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x_id = x
        x = self.silu(self.bn(self.conv_1(x)))
        x = self.conv_2(x)
        x_id = self.mapping(x_id) if self.mapping else x_id
        x = self.silu(self.bn(x + x_id))
        return x


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), neck=4):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        middle_channel = math.ceil(out_channels/neck)
        self.conv_1 = nn.Conv3d(in_channels, middle_channel, kernel_size=1, bias=False)
        self.conv_2 = nn.Conv3d(middle_channel, middle_channel, kernel_size=kernel_size,
                                padding=padding, padding_mode='zeros', bias=False)
        self.conv_3 = nn.Conv3d(middle_channel, out_channels, kernel_size=1)
        self.bn1 = nn.InstanceNorm3d(middle_channel, track_running_stats=False)
        self.bn2 = nn.InstanceNorm3d(out_channels, track_running_stats=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x_id = x
        x = self.silu(self.bn1(self.conv_1(x)))
        x = self.silu(self.bn1(self.conv_2(x)))
        x = self.conv_3(x)
        x_id = self.mapping(x_id) if self.mapping else x_id
        x = self.silu(self.bn2(x + x_id))
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv_1 = nn.Conv3d(in_channels, out_channels,
                                kernel_size=kernel_size, padding=padding, padding_mode='zeros', bias=False)
        self.conv_2 = nn.Conv3d(out_channels, out_channels,
                                kernel_size=kernel_size, padding=padding, padding_mode='zeros', bias=False)
        self.bn = nn.InstanceNorm3d(out_channels, track_running_stats=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.silu(self.bn(self.conv_1(x)))
        x = self.silu(self.bn(self.conv_2(x)))
        return x


class BasicBlockSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv_1 = nn.Conv3d(in_channels, out_channels,
                                kernel_size=kernel_size, padding=padding, padding_mode='zeros', bias=False)
        self.bn = nn.InstanceNorm3d(out_channels, track_running_stats=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.silu(self.bn(self.conv_1(x)))
        return x


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.silu = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 2, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        x_avg = self.avg_pool(x).view(batch_size, num_channels)
        x_avg = self.silu(self.fc1(x_avg))
        x_avg = self.sigmoid(self.fc2(x_avg))
        return x * (x_avg.view(batch_size, num_channels, 1, 1, 1))


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_spatial = self.fc(x)
        x_spatial = self.sigmoid(x_spatial)
        return x * x_spatial


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cse = cSE(in_channels)
        self.sse = sSE(in_channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


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
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature_maps):
        # feature_maps is a list of features from different resolution branches
        num_features = len(feature_maps)
        fused_features = [feature_maps[i] for i in range(num_features)]  # Start with the original feature maps

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

        # Apply activation after fusion
        fused_features = [self.activation(f) for f in fused_features]

        return fused_features