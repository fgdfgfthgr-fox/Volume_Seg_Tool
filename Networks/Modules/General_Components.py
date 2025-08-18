import math
import numpy as np
import torch
import torch.nn as nn


class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), num_conv=2, norm=True):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        layers = []
        if in_channels == out_channels:
            self.mapping = nn.Identity()
        else:
            self.mapping = nn.Conv3d(in_channels, out_channels, 1, padding)
        for i in range(num_conv):
            if norm:
                layers.append(nn.InstanceNorm3d(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
            bias = False if i != num_conv - 1 else True
            layers.append(nn.Conv3d(in_channels if i == 0 else out_channels, out_channels,
                                    kernel_size=kernel_size, padding=padding, bias=bias))
        self.operations = nn.Sequential(*layers)
        #self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.mapping(x) + self.operations(x)



class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), neck=4, num_conv=1, norm=True):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        middle_channel = out_channels//neck
        self.conv_down = nn.Conv3d(in_channels, middle_channel, kernel_size=1, bias=False)

        layers = []
        for i in range(num_conv):
            layers.append(
                nn.Conv3d(middle_channel, middle_channel, kernel_size=kernel_size, padding=padding, bias=False))
            if norm:
                layers.append(nn.InstanceNorm3d(out_channels))
            if i != num_conv - 1:
                layers.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

        self.conv_up = nn.Conv3d(middle_channel, out_channels, kernel_size=1)
        if norm:
            self.bn = nn.InstanceNorm3d(middle_channel)
        else:
            self.bn = nn.Identity()
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        b_x = self.act(self.bn(self.conv_down(x)))
        b_x = self.convs(b_x)
        b_x = self.conv_up(b_x)
        return x + b_x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), num_conv=2, padding_mode='zeros', norm=True):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        layers = []
        for i in range(num_conv):
            layers.append(nn.Conv3d(in_channels if i == 0 else out_channels, out_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False, padding_mode=padding_mode))
            if norm:
                layers.append(nn.InstanceNorm3d(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
        self.operations = nn.Sequential(*layers)

    def forward(self, x):
        return self.operations(x)


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, in_channels)
        #self.act = nn.LeakyReLU(inplace=True)
        self.glu = nn.GLU()
        self.fc2 = nn.Linear(in_channels // 2, in_channels)

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        x_avg = self.avg_pool(x).view(batch_size, num_channels)
        x_avg = self.fc1(x_avg)
        x_avg = self.glu(x_avg)
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


class FourierShells(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, x):
        B, C, D, H, W = x.shape

        # Compute RFFT (optimized for real inputs)
        x = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='ortho')  # Output: [B, 1, D, H, W//2+1]

        # Create frequency coordinates for half-spectrum
        d_coord = torch.fft.fftfreq(D, device=x.device, dtype=torch.float16) * D
        h_coord = torch.fft.fftfreq(H, device=x.device, dtype=torch.float16) * H
        w_coord = torch.fft.rfftfreq(W, device=x.device, dtype=torch.float16) * W  # Only non-negative freqs

        # Compute radial distances (only for non-redundant components)
        r = torch.sqrt(
            (d_coord.view(-1, 1, 1) ** 2) +
            (h_coord.view(1, -1, 1) ** 2) +
            (w_coord.view(1, 1, -1) ** 2))

        # Logarithmic shell boundaries
        boundaries = torch.logspace(
            0,
            torch.log2(r.max()),
            self.K + 1,
            base=2,
            device=x.device,
            dtype=torch.float16
        )
        boundaries[0] = 0
        # Generate radial masks
        masks = []
        for k in range(self.K):
            mask_k = (r >= boundaries[k]) & (r <= boundaries[k + 1])
            masks.append(mask_k)

        # Apply masks and compute inverse RFFT
        masks_tensor = torch.stack(masks)  # [K, D, H, W//2+1]
        shells = torch.fft.irfftn(
            x * masks_tensor,
            s=(D, H, W),  # Specify full output shape
            dim=(-3, -2, -1),
            norm='ortho'
        )  # Output: [B, K, D, H, W]

        return shells


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


class DyT(nn.Module):
    def __init__(self, num_features, dim=1):
        """
        Dynamic Tanh (DyT) layer

        Args:
            num_features (int): Number of features/channels (same as corresponding normalization layer)
            dim (int, optional): Dimension along which to apply parameters.
                                Default: 1 (channel dimension for Conv layers)
        """
        super().__init__()
        self.num_features = num_features
        self.dim = dim

        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))  # scalar
        self.gamma = nn.Parameter(torch.ones(num_features))  # per-channel vector
        self.beta = nn.Parameter(torch.zeros(num_features))  # per-channel vector

    def forward(self, x):
        # Reshape gamma and beta to match input dimensions
        shape = [1] * x.ndim
        shape[self.dim] = self.num_features
        gamma = self.gamma.view(*shape)
        beta = self.beta.view(*shape)

        # Apply dynamic tanh transformation
        return gamma * torch.tanh(self.alpha * x) + beta