import torch
import torch.nn as nn
from .Modules import General_Components as Modules

class MicroTNet(nn.Module):

    def __init__(self, base_channels=8):
        super(MicroTNet, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm3d(base_channels, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, padding_mode="replicate"),
        )
        self.down = nn.MaxPool3d(2)
        self.bottleneck = Modules.DoubleConv3D(base_channels, 2 * base_channels, bn=True)
        self.decode1 = Modules.Up3D(2 * base_channels, base_channels, bn=True)

        self.out = nn.Sequential(nn.Conv3d(base_channels, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = self.inc(x)
        x_1 = x
        x = self.down(x)
        x = self.bottleneck(x)
        x = self.decode1(x, x_1)
        x = self.out(x)
        return x