import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels, track_running_stats=False)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels, track_running_stats=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
