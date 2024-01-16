import torch
import torch.nn as nn
import torch.nn.functional as F


# Two convolutions bundled together with BN and ReLU, commonly used in UNet and SegNet
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1), bn=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                   (kernel_size[1] - 1) * dilation[1] // 2,
                   (kernel_size[2] - 1) * dilation[2] // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, padding_mode='replicate',
                               dilation=dilation, bias=False)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, stride=stride,
                               kernel_size=kernel_size, padding=padding, padding_mode='replicate',
                               dilation=dilation, bias=False)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


# Down-sample the volume, then perform a DoubleConv3D. Support Max Pool and Avg Pool.
class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), pool_kernel_size=(2, 2, 2), method="max-pool", dilation=(1, 1, 1)):
        super().__init__()
        if method == "max-pool":
            self.steps = nn.Sequential(
                nn.MaxPool3d(
                    kernel_size=pool_kernel_size,
                    ceil_mode=True,
                ),
                DoubleConv3D(in_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn)
            )
        elif method == "avg-pool":
            self.steps = nn.Sequential(
                nn.AvgPool3d(
                    kernel_size=pool_kernel_size,
                    ceil_mode=True,
                ),
                DoubleConv3D(in_channels, out_channels, conv_kernel_size, dilation=dilation, bn=bn)
            )

    def forward(self, x):
        return self.steps(x)


# Upscale the volume using transposed convolution or interpolation, then perform a DoubleConv3D.
class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True,
                 conv_kernel_size=(3, 3, 3), scale_kernel_size=(2, 2, 2), method="transpose"):
        super().__init__()
        if method == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=in_channels // 2,
                kernel_size=scale_kernel_size, stride=scale_kernel_size
            )
        self.method = method
        self.scale_kernel_size = scale_kernel_size
        self.conv = DoubleConv3D(in_channels, out_channels, conv_kernel_size, bn=bn)

    def forward(self, x1, x2):
        if self.method == "transpose":
            x1 = self.up(x1)
        elif self.method == "interpolation":
            x1 = F.interpolate(x1, scale_factor=self.scale_kernel_size, mode="trilinear", align_corners=True)
        # 图像：(N,C,D,H,W) N=Batch C=Channel
        # x1 is always equal or larger than x2
        diffD = x1.size()[-3] - x2.size()[-3]
        diffH = x1.size()[-2] - x2.size()[-2]
        diffW = x1.size()[-1] - x2.size()[-1]
        if not diffD == 0:
            x2 = x2[:, :, :-diffD, :, :]
        if not diffH == 0:
            x2 = x2[:, :, :, :-diffH, :]
        if not diffW == 0:
            x2 = x2[:, :, :, :, :-diffW]
        # Concatenates the channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Merge the result from different layers by simply adding them together, used for Half-UNet
class Merge3D(nn.Module):
    def __init__(self, scale):
        super(Merge3D, self).__init__()
        self.scaleup = torch.nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True)

    def forward(self, x1, x2):
        x2 = self.scaleup(x2)
        diffD = x2.size()[-3] - x1.size()[-3]
        diffH = x2.size()[-2] - x1.size()[-2]
        diffW = x2.size()[-1] - x1.size()[-1]
        if not diffD == 0:
            x2 = x2[:, :, :-diffD, :, :]
        if not diffH == 0:
            x2 = x2[:, :, :, :-diffH, :]
        if not diffW == 0:
            x2 = x2[:, :, :, :, :-diffW]
        return torch.add(x1, x2)


def UpsampleAndAdd(input_tensors, scale_factors, mode="crop"):
    target_shape = input_tensors[0].shape[-3:]
    if mode == "crop":
        scaled_tensors = [F.interpolate(tensor, scale_factor=scale, mode='trilinear', align_corners=True) for
                          tensor, scale
                          in zip(input_tensors, scale_factors)]
        cropped_tensors = [tensor[:, :, :-d, :, :] if d != 0 else tensor for tensor, d in
                           zip(scaled_tensors, [tensor.shape[-3] - target_shape[-3] for tensor in scaled_tensors])]
        cropped_tensors = [tensor[:, :, :, :-h, :] if h != 0 else tensor for tensor, h in
                           zip(cropped_tensors, [tensor.shape[-2] - target_shape[-2] for tensor in cropped_tensors])]
        cropped_tensors = [tensor[:, :, :, :, :-w] if w != 0 else tensor for tensor, w in
                           zip(cropped_tensors, [tensor.shape[-1] - target_shape[-1] for tensor in cropped_tensors])]
    if mode == "stretch":
        cropped_tensors = [F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=True) for
                           tensor, scale in zip(input_tensors, scale_factors)]
    result = torch.stack(cropped_tensors).sum(dim=0)
    return result