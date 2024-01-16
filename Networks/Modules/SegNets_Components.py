import torch.nn as nn


class SegEncodeBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels,
                 conv_kernel_size=(7, 7, 7), pool_kernel_size=(2, 2, 2)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in conv_kernel_size)
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size, ceil_mode=True, return_indices=True)

    def forward(self, x):
        output, indices = self.pool(self.steps(x))
        return output, indices


class SegEncodeBlock_3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 conv_kernel_size=(7, 7, 7), pool_kernel_size=(2, 2, 2)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in conv_kernel_size)
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size, ceil_mode=True, return_indices=True)

    def forward(self, x):
        output, indices = self.pool(self.steps(x))
        return output, indices


class SegDecodeBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels,
                 conv_kernel_size=(7, 7, 7), unpool_kernel_size=(2, 2, 2)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in conv_kernel_size)
        self.unpool = nn.MaxUnpool3d(kernel_size=unpool_kernel_size)
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, indices):
        output = self.steps(self.unpool(x, indices))
        return output


class SegDecodeBlock_3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 conv_kernel_size=(7, 7, 7), unpool_kernel_size=(2, 2, 2)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in conv_kernel_size)
        self.unpool = nn.MaxUnpool3d(kernel_size=unpool_kernel_size)
        self.steps = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, indices):
        output = self.steps(self.unpool(x, indices))
        return output