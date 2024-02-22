import torch.nn as nn
from .Modules.SegNets_Components import SegEncodeBlock_2, SegEncodeBlock_3, SegDecodeBlock_2, SegDecodeBlock_3
from .Modules.General_Components import scSE
import math


# Badrinarayanan V, Kendall A, Cipolla R. Segnet: A deep convolutional encoder-decoder architecture for image
# segmentation[J]. IEEE transactions on pattern analysis and machine intelligence, 2017, 39(12): 2481-2495.

# Use max pooling indices to restore spatial information, hence no skip connections used.

# What I implemented here is actually similar to the "SegNet-Basic" in the paper.
# The default kernel size for all conv layers is 3 instead of 7 (in the paper) due to the size of 7 take too much resources in the 3D cases.
# Original: The "Original" Architecture as proposed in paper, except all operations changed to their 3D equivalent.
# Auto: My personal touch on automatic generated network architecture based on the required depth.


class Original(nn.Module):
    def __init__(self, base_channels=64, kernel_size=(3, 3, 3)):
        super(Original, self).__init__()
        self.encode1 = SegEncodeBlock_2(1, base_channels, conv_kernel_size=kernel_size)
        self.encode2 = SegEncodeBlock_2(base_channels, base_channels * 2, conv_kernel_size=kernel_size)
        self.encode3 = SegEncodeBlock_3(base_channels * 2, base_channels * 4, conv_kernel_size=kernel_size)
        self.encode4 = SegEncodeBlock_3(base_channels * 4, base_channels * 8, conv_kernel_size=kernel_size)
        self.encode5 = SegEncodeBlock_3(base_channels * 8, base_channels * 16, conv_kernel_size=kernel_size)

        self.decode1 = SegDecodeBlock_3(base_channels * 16, base_channels * 8, conv_kernel_size=kernel_size)
        self.decode2 = SegDecodeBlock_3(base_channels * 8, base_channels * 4, conv_kernel_size=kernel_size)
        self.decode3 = SegDecodeBlock_3(base_channels * 4, base_channels * 2, conv_kernel_size=kernel_size)
        self.decode4 = SegDecodeBlock_2(base_channels * 2, base_channels, conv_kernel_size=kernel_size)
        self.decode5 = SegDecodeBlock_2(base_channels, 2, conv_kernel_size=kernel_size)
        # modified output layer to suit the binary classification task
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        x, indice1 = self.encode1(input)
        x, indice2 = self.encode2(x)
        x, indice3 = self.encode3(x)
        x, indice4 = self.encode4(x)
        x, indice5 = self.encode5(x)

        x = self.decode1(x, indice5)
        x = self.decode2(x, indice4)
        x = self.decode3(x, indice3)
        x = self.decode4(x, indice2)
        x = self.decode5(x, indice1)

        output = self.final(x)
        return output


class Auto(nn.Module):
    def __init__(self, base_channels=64, depth=5, z_to_xy_ratio=1, se=False):
        super(Auto, self).__init__()
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        depth = depth - 1 # The way I wrote SegNet modules means it's actually always one layer more than specified
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        self.se = se
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio require deeper depth to make network effective.")
        kernel_sizes_conv = []
        kernel_sizes_pool = []
        for i in range(depth):
            if self.special_layers > 0 and i < self.special_layers:
                kernel_sizes_conv.append((1, 3, 3))
                kernel_sizes_pool.append((1, 2, 2))
            elif self.special_layers < 0 and i < -self.special_layers:
                kernel_sizes_conv.append((3, 1, 1))
                kernel_sizes_pool.append((2, 1, 1))
            else:
                kernel_sizes_conv.append((3, 3, 3))
                kernel_sizes_pool.append((2, 2, 2))
        for i in range(depth):
            if i == 0:
                setattr(self, f'encode0',
                        SegEncodeBlock_2(1, base_channels, kernel_sizes_conv[0], kernel_sizes_pool[0]))
                if se: setattr(self, f'encode_se0', scSE(base_channels))
                setattr(self, f'decode0',
                        SegDecodeBlock_2(base_channels, 2, kernel_sizes_conv[0], kernel_sizes_pool[0]))
                if se: setattr(self, f'decode_se0', scSE(2))
            else:
                setattr(self, f'encode{i}',
                        SegEncodeBlock_3(base_channels * (2 ** (i-1)), (base_channels * (2 ** i)),
                                         kernel_sizes_conv[i], kernel_sizes_pool[i]))
                if se: setattr(self, f'encode_se{i}', scSE(base_channels * (2 ** i)))
                setattr(self, f'decode{i}',
                        SegDecodeBlock_3(base_channels * (2 ** i), (base_channels * (2 ** (i-1))),
                                         kernel_sizes_conv[i], kernel_sizes_pool[i]))
                if se: setattr(self, f'decode_se{i}', scSE(base_channels * (2 ** (i-1))))
        self.final = nn.Sequential(nn.Conv3d(2, 1, kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, input):
        encode_indices = []
        x = input

        for i in range(self.depth):
            x, indices = getattr(self, f"encode{i}")(x)
            if self.se: x = getattr(self, f"encode_se{i}")(x)
            encode_indices.append(indices)

        for i in reversed(range(self.depth)):
            x = getattr(self, f"decode{i}")(x, encode_indices[i])
            if self.se: x = getattr(self, f"decode_se{i}")(x)

        output = self.final(x)

        return output
