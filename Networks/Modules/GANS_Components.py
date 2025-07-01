import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class ConstantInput(nn.Module):
    def __init__(self, channel, size=(4, 4, 4), precision=torch.float32):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, *size, dtype=precision))

    def forward(self, batch_size):
        out = self.input.repeat(batch_size, 1, 1, 1, 1)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + torch.finfo(input.dtype).eps)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bias_init=0, lr_mul=1.0, precision=torch.float32):
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
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=precision).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=precision).fill_(bias_init))
        else:
            self.register_parameter('bias', None)

        # He's scaling factor
        self.scale = (1 / math.sqrt(in_features)) * lr_mul

    def forward(self, x):
        x = F.linear(x, self.weight * self.scale, self.bias)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, lr_mul={self.lr_mul}'


def create_gaussian_kernel_3d(kernel_size=3, sigma=1.0):
    """Creates a 3D Gaussian kernel."""
    # Create a 3D grid
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')

    # Compute Gaussian
    kernel = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2. * sigma ** 2))
    kernel = kernel / kernel.sum()  # Normalize
    return kernel

class LowPassFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        k = create_gaussian_kernel_3d(3, 1)
        k = k.expand(channels, 1, 3, 3, 3).clone()
        #self.register_parameter('k', k)
        self.weight = torch.nn.parameter.Buffer(k)

    def forward(self, x):
        weight = self.weight.to(x.dtype)
        x = F.conv3d(x, weight=weight, padding='same', groups=self.channels)
        return x


class ModulatedConv3d(nn.Module):
    """3D modulated convolution (StyleGAN2's weight modulation/demodulation)."""

    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=False, precision=torch.float32):
        super().__init__()
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.upsample = upsample
        # Only used when upsample = True
        self.upscale_factor = tuple((k+1)//2 for k in kernel_size)
        self.padding = tuple(k//2-1 for k in kernel_size)

        #self.blur = LowPassFilter(out_channels, precision)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2], dtype=precision))
        self.scale = 1 / math.sqrt(in_channels * (kernel_size[0]*kernel_size[1]*kernel_size[2]))
        self.modulation = EqualizedLinear(style_dim, in_channels, bias_init=1, precision=precision)

    def forward(self, x, style):
        batch, in_channels, depth, height, width = x.shape

        weight = self.scale * self.weight.squeeze(0)
        style = self.modulation(style)

        if self.demodulate:
            w = weight.unsqueeze(0) * style.view(batch, 1, in_channels, 1, 1, 1)
            dcoefs = (w.square().sum((2, 3, 4, 5)) + 1e-8).rsqrt().to(weight.dtype)

        x = x * style.reshape(batch, in_channels, 1, 1, 1)

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='trilinear')
        x = F.conv3d(x, weight, padding='same')

        if self.demodulate:
            x = x * dcoefs.view(batch, -1, 1, 1, 1)

        return x


class NoiseInjection(nn.Module):
    def __init__(self, precision=torch.float32):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, dtype=precision))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, depth, height, width = image.shape
            #noise = image.new_empty(batch, 1, depth, height, width).normal_()
            noise = torch.randn((batch, 1, depth, height, width), device=image.device, dtype=image.dtype)

        return image + self.weight * noise


class StyledConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=False, precision=torch.float32):
        super().__init__()

        self.conv = ModulatedConv3d(
            in_channels,
            out_channels,
            kernel_size,
            style_dim,
            demodulate=demodulate,
            upsample=upsample,
            precision=precision
        )

        self.noise = NoiseInjection(precision)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1, dtype=precision))
        self.activate = torch.nn.LeakyReLU(0.2)
        # self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = out + self.bias
        out = self.activate(out)

        return out


class ToG(nn.Module):
    def __init__(self, in_channel, style_dim, precision=torch.float32):
        super().__init__()

        self.conv = ModulatedConv3d(in_channel, 1, (1,1,1), style_dim, demodulate=False, precision=precision)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1, dtype=precision))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = F.interpolate(skip, size=out.shape[-3:], mode='trilinear', align_corners=False).to(out.dtype)
            out = out + skip

        return out


class MinibatchStd3d(nn.Module):
    """Minibatch standard deviation for 3D (StyleGAN2)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        std = torch.std(x, dim=(0, 1), keepdim=True) + torch.finfo(x.dtype).eps
        std = std.to(x.dtype)
        x = torch.cat([x, std.expand(x.shape[0], 1, *x.shape[2:])], dim=1)
        return x


class SimpleGANConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample=False):
        super().__init__()

        self.upsample = upsample
        self.upscale_factor = tuple(math.ceil(k/2) for k in kernel_size)
        self.upscale_unpadding = tuple((k-2)//2 for k in kernel_size)
        #if self.upsample:
        #    self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=self.upscale_factor, bias=False)
        #else:
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding='same', bias=False, padding_mode='reflect')
        self.blur = LowPassFilter(in_channels)
        self.noise_factors = nn.Parameter(torch.ones(1, in_channels, 1, 1, 1)*0.1)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activate = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        batch, channel, depth, height, width = x.shape
        noise = torch.randn(batch, channel, depth, height, width, dtype=x.dtype, device=x.device, requires_grad=True)
        noise = self.noise_factors.to(x.dtype) * noise
        x = x + noise
        if self.upsample:
            x = F.interpolate(x, scale_factor=self.upscale_factor, mode='nearest')
            x = self.blur(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)
        return x

class SimpleModulatedStyleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=False, precision=torch.float32):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = upsample
        # Only used when upsample = True
        self.upscale_factor = tuple(math.ceil(k / 2) for k in kernel_size)
        self.upscale_unpadding = tuple((k - 2) // 2 for k in kernel_size)

        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2], dtype=precision))
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1, dtype=precision))
        self.scale = 1 / math.sqrt(in_channels * (kernel_size[0]*kernel_size[1]*kernel_size[2]))
        self.activate = torch.nn.SELU(inplace=True)

    def forward(self, x):
        batch, in_channels, depth, height, width = x.shape

        weight = self.scale * self.weight.squeeze(0)

        w = weight.unsqueeze(0).expand(batch, -1, -1, -1, -1, -1)
        dcoefs = (w.square().sum((2, 3, 4, 5)) + 1e-8).rsqrt().to(weight.dtype)

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='nearest')
        x = F.conv3d(x, weight, padding='same')
        dcoefs = dcoefs.view(batch, -1, 1, 1, 1)
        x = x * dcoefs
        x += self.bias

        x = self.activate(x)

        return x




class SimpleToG(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.conv = nn.Conv3d(in_channel, 1, kernel_size=1)

    def forward(self, input, skip=None):
        out = self.conv(input)

        if skip is not None:
            skip = F.interpolate(skip, size=out.shape[-3:], mode='trilinear', align_corners=False)
            out = out + skip

        return out


class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), num_conv=2):
        super().__init__()
        layers = []
        if in_channels == out_channels:
            self.mapping = nn.Identity()
        else:
            self.mapping = nn.Conv3d(in_channels, out_channels, 1, padding='same')
        for i in range(num_conv):
            layers.append(
                    (nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel_size, padding='same', padding_mode='reflect', bias=False)))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.SELU(inplace=True))
        self.operations = nn.Sequential(*layers)

    def forward(self, x):
        return (self.mapping(x) + self.operations(x)) / math.sqrt(2)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), num_conv=2):
        super().__init__()
        layers = []
        for i in range(num_conv):
            layers.append(
                    (nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel_size, padding='same', padding_mode='reflect', bias=False)))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.SELU(inplace=True))
        self.operations = nn.Sequential(*layers)

    def forward(self, x):
        return self.operations(x)