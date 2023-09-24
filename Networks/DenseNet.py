import torch
import torch.nn as nn
from .Modules import DenseNet_Components as DenseModules


# Jégou, S., Drozdzal, M., Vazquez, D., Romero, A., Bengio, Y.: The one hundred layers tiramisu: fully convolutional
# densenets for semantic segmentation. In: CVPR Workshop, pp. 1175–1183. IEEE, July 2017

# The original DenseNet is for image classification, but in the paper above it's converted to an sematic segmentation network.
# Code modified from: https://github.com/bfortuner/pytorch_tiramisu/tree/master


class FCDenseNet(nn.Module):
    def __init__(self, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, dropout_p=0.2):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv3d(in_channels=1,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseModules.DenseBlock(cur_channels_count, growth_rate, dropout_p, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(DenseModules.TransDown(cur_channels_count, dropout_p))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', DenseModules.Bottleneck(cur_channels_count,
                                     growth_rate, dropout_p, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(DenseModules.TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseModules.DenseBlock(
                cur_channels_count, growth_rate, dropout_p, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(DenseModules.TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseModules.DenseBlock(
            cur_channels_count, growth_rate, dropout_p, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv3d(in_channels=cur_channels_count,
                                   out_channels=1, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.sigmoid(out)
        return out


def FCDenseNet57(growth_rate=12, base_channels=48, dropout_p=0.2):
    return FCDenseNet(
        down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=growth_rate, out_chans_first_conv=base_channels,
        dropout_p=dropout_p)


def FCDenseNet67(growth_rate=16, base_channels=48, dropout_p=0.2):
    return FCDenseNet(
        down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=growth_rate, out_chans_first_conv=base_channels,
        dropout_p=dropout_p)


def FCDenseNet103(growth_rate=16, base_channels=48, dropout_p=0.2):
    return FCDenseNet(
        down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=growth_rate, out_chans_first_conv=base_channels,
        dropout_p=dropout_p)