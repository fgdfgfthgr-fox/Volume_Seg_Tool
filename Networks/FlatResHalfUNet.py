import torch
import torch.nn as nn
from .Modules import General_Components as Modules


class FlatResHalfUNet(nn.Module):

    def __init__(self, base_channels=4, depth=3, in_channel=1, out_channel=2):
        super(FlatResHalfUNet, self).__init__()
        self.depth = depth
        self.inc = nn.Sequential(
            nn.Conv3d(in_channel, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="replicate"),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="replicate"),
        )
        self.inc_skip = nn.Conv3d(1, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="replicate")

        self.conv1 = Modules.ResidualConv3D(base_channels, base_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.merge1 = Modules.Merge3D((1, 2, 2))
        if self.depth >= 2:
            for i in range(2, depth):
                setattr(self, f'conv{i}', Modules.ResidualConv3D(base_channels, base_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2)))
                setattr(self, f'merge{i}', Modules.Merge3D((1, 2 ** i, 2 ** i)))

        self.res_z_work = Modules.ResidualConv3D(base_channels, base_channels, kernel_size=(3, 3, 3))
        self.outc = torch.nn.Linear(base_channels, out_channel)

    def forward(self, x):
        x = self.inc(x) + self.inc_skip(x)
        x_1 = self.conv1(x)
        x = self.merge1(x, x_1)
        if self.depth >= 2:
            for i in range(2, self.depth):
                x_1 = getattr(self, f'conv{i}')(x_1)
                x = getattr(self, f'merge{i}')(x, x_1)
        x = self.res_z_work(x)
        x = torch.permute(x, (0,4,2,3,1))
        x = self.outc(x)
        x = torch.permute(x, (0,4,2,3,1))
        return x