import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstantInput(nn.Module):
    def __init__(self, channel, size=(4, 4, 4)):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, *size))

    def forward(self, batch_size):
        out = self.input.repeat(batch_size, 1, 1, 1, 1)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bias_init=0, lr_mul=1.0):
        """
        Equalized learning rate linear layer as used in StyleGAN2.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias term
            bias_init: Initial value for bias (0 in StyleGAN2)
            lr_mul: Learning rate multiplier (for mapping network in StyleGAN2)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_mul = lr_mul

        # Initialize weights with normal distribution
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.bias_init = bias_init
        else:
            self.register_parameter('bias', None)

        # He's scaling factor
        self.scale = (1 / math.sqrt(in_features)) * lr_mul
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with standard deviation of 1.0
        nn.init.normal_(self.weight, mean=0, std=1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, self.bias_init)

    def forward(self, x):
        # Scale weights and apply linear operation
        weight = self.weight * self.scale
        if self.bias is not None:
            bias = self.bias * self.lr_mul
        else:
            bias = None
        return F.linear(x, weight, bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, lr_mul={self.lr_mul}'


class ModulatedConv3d(nn.Module):
    """3D modulated convolution (StyleGAN2's weight modulation/demodulation)."""

    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=True):
        super().__init__()
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.upsample = upsample
        self.upscale_factor = tuple((k+1)//2 for k in kernel_size)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]))
        self.scale = 1 / math.sqrt(in_channels * (kernel_size[0]*kernel_size[1]*kernel_size[2]))  # He init scaling
        self.modulation = EqualizedLinear(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channels, depth, height, width = x.shape

        weight = self.scale * self.weight.squeeze(0)
        style = self.modulation(style)

        if self.demodulate:
            w = weight.unsqueeze(0) * style.view(batch, 1, in_channels, 1, 1, 1)
            dcoefs = (w.square().sum((2, 3, 4, 5)) + 1e-8).rsqrt()

        x = x * style.reshape(batch, in_channels, 1, 1, 1)

        if self.upsample:
            x = F.interpolate(x, scale_factor=self.upscale_factor, mode='trilinear')

        x = F.conv3d(x, weight, padding='same')

        if self.demodulate:
            x = x * dcoefs.view(batch, -1, 1, 1, 1)

        return x


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, depth, height, width = image.shape
            noise = image.new_empty(batch, 1, depth, height, width).normal_()

        return image + self.weight * noise


class StyledConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=False):
        super().__init__()

        self.conv = ModulatedConv3d(
            in_channels,
            out_channels,
            kernel_size,
            style_dim,
            demodulate=demodulate,
            upsample=upsample,
        )

        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1))
        self.activate = torch.nn.LeakyReLU(0.2)
        # self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = out + self.bias
        out = self.activate(out)

        return out


class ToG(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True):
        super().__init__()

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)

        self.conv = ModulatedConv3d(in_channel, 1, (1,1,1), style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class MinibatchStd3d(nn.Module):
    """Minibatch standard deviation for 3D (StyleGAN2)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        std = torch.std(x, dim=0).mean()
        return torch.cat([x, std.expand(x.shape[0], 1, *x.shape[2:])], dim=1)