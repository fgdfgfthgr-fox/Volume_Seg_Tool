import torch
import torch.nn as nn
import torch.nn.functional as F

class TLU(nn.Module):  # Trainable Linear Unit (TLU)
    def __init__(self, num_channels, alpha_init=0.0, beta_init=1.0):
        """
        Initialize the TLU with trainable parameters alpha and beta.

        Args:
            num_channels (int): Number of channels in the input.
            alpha_init (float): Initial value for the alpha parameter.
            beta_init (float): Initial value for the beta parameter.
        """
        super(TLU, self).__init__()

        # Create separate alpha and beta parameters for each channel
        self.alpha = nn.Parameter(torch.tensor([alpha_init] * num_channels))
        self.beta = nn.Parameter(torch.tensor([beta_init] * num_channels))

    def forward(self, x):
        # Get the number of dimensions in the input (2D or 3D)
        num_dims = len(x.shape) - 2

        # Reshape alpha and beta for broadcasting
        alpha_reshaped = self.alpha.view(1, -1, *(1 for _ in range(num_dims)))
        beta_reshaped = self.beta.view(1, -1, *(1 for _ in range(num_dims)))

        # Apply the TLU activation with dynamic broadcasting
        output = torch.where(x >= 0, beta_reshaped * x, alpha_reshaped * x)
        return output

class EMNetBaseModule_3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), bn=False):
        super(EMNetBaseModule_3D, self).__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, padding_mode='replicate')
        self.tlu = TLU(out_channels)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.tlu(x)

        return x


class EMNetCell1_3D(nn.Module):
    # Orange cells, three layers of convolutions each having the same number of feature channels where first and third layers are merged to form the output of the cell
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), bn=False):
        super(EMNetCell1_3D, self).__init__()
        self.conv_block_1 = EMNetBaseModule_3D(in_channels, out_channels // 2, kernel_size, stride, bn=bn)
        self.conv_block_2 = EMNetBaseModule_3D(out_channels // 2, out_channels // 2, kernel_size, stride, bn=bn)
        self.conv_block_3 = EMNetBaseModule_3D(out_channels // 2, out_channels // 2, kernel_size, stride, bn=bn)

    def forward(self, x):
        x = self.conv_block_1(x)
        x_id = x
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = torch.cat([x, x_id], dim=1)
        return x


class EMNetCell2_3D(nn.Module):
    # Purple cells, four layers of convolutions, where the odd layers have twice the number of feature channels of the even layers
    # Moreover, the second and fourth layers are merged to form the output in these cells
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), bn=False):
        super(EMNetCell2_3D, self).__init__()
        self.conv_block_1 = EMNetBaseModule_3D(in_channels, out_channels, kernel_size, stride, bn=bn)
        self.conv_block_2 = EMNetBaseModule_3D(out_channels, out_channels // 2, kernel_size, stride, bn=bn)
        self.conv_block_3 = EMNetBaseModule_3D(out_channels // 2, out_channels, kernel_size, stride, bn=bn)
        self.conv_block_4 = EMNetBaseModule_3D(out_channels, out_channels // 2, kernel_size, stride, bn=bn)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x_id = x
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = torch.cat([x, x_id], dim=1)
        return x


class EMNetUp_3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(1, 1, 1), bn=True):
        super(EMNetUp_3D, self).__init__()
        self.scaleup = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=0, padding_mode='replicate')
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.scaleup(x)
        x = F.pad(x, [0, 1, 0, 1, 0, 1], mode='replicate')
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        return x

class EMNetCellV24X_3D(nn.Module):
    # The special EM cell structure used in v2 4x model.
    # The actual out channels are 4x the out_channels arg, because it's how they implemented it...
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), bn=False):
        super(EMNetCellV24X_3D, self).__init__()
        self.conv_block_1 = EMNetBaseModule_3D(in_channels, out_channels * 2, kernel_size, stride, bn=bn)
        self.conv_block_2 = EMNetBaseModule_3D(out_channels * 2, out_channels, kernel_size, stride, bn=bn)
        self.conv_block_3 = EMNetBaseModule_3D(out_channels * 3, out_channels, kernel_size, stride, bn=bn)

    def forward(self, x):
        x = self.conv_block_1(x)
        x_1 = self.conv_block_2(x)
        x = torch.cat([x, x_1], dim=1)
        x_1 = x
        x = self.conv_block_3(x)
        x = torch.cat([x, x_1], dim=1)
        return x

class EMNetCellV24X_3D_MID(nn.Module):
    # The special EM cell structure used in v2 4x model.
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super(EMNetCellV24X_3D_MID, self).__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode='replicate')
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode='replicate')

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x