import torch
import torch.nn as nn
import math
from .Modules.General_Components import ResBasicBlock, ResBottleneckBlock, BasicBlock, scSE

# Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
# In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich,
# Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

# Çiçek Ö, Abdulkadir A, Lienkamp S S, et al. 3D U-Net: learning dense volumetric segmentation from sparse
# annotation[C]//Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International
# Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19. Springer International Publishing,
# 2016: 424-432.

# This file contains U-net-like Architectures for instance segmentation,
# with the main difference being the skip connection adds the channels instead of concatenate them
# Two branches, p(ixel) branch predict the mask, which c(ontour) branch predict contour of objects
# The two branches shares the same encoder
# Although later experiments has confirmed that just using two different final convolution is sufficient.
# Basic: Use normal convolution block
# Residual: Use residual block
# ResidualBottleneck: Use residual bottleneck block

# U-net: The most influential DL image classification paper.
# Introduced the concept of skip connection to feed spatial information into the decoder part of the network
# The original UNet implementation in the paper doesn't use batch normalization, nor does conv padding

# Optionally can have squeeze and excite blocks after each conv block.
# https://link.springer.com/chapter/10.1007/978-3-030-00928-1_48
# Could be considered as an attention mechanism which boost meaningful features, while suppressing weak ones.


class BranchedUNet(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic',
                 se=False, label_mean=torch.tensor(0.5), contour_mean=torch.tensor(0.5)):
        super(BranchedUNet, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        self.se = se
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio requires a deeper network.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = (3, 3, 3)

        kernel_sizes_transpose = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                                  (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                  (2, 2, 2) for i in range(depth)]
        kernel_sizes_down = [(1, 4, 4) if self.special_layers > 0 and i < self.special_layers else
                             (4, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                             (4, 4, 4) for i in range(depth)]
        padding_down = [(0, 1, 1) if self.special_layers > 0 and i < self.special_layers else
                        (1, 0, 0) if self.special_layers < 0 and i < -self.special_layers else
                        (1, 1, 1) for i in range(depth)]
        block = {'Basic': BasicBlock, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        for i in range(depth):
            if i == 0:
                num_conv = 1
            else:
                num_conv = 2
            multiplier_h = base_channels * (2 ** i)
            multiplier_v = base_channels * (2 ** (i+1))
            if i != depth - 1:
                if i == 0:
                    if type == 'ResidualBottleneck':
                        multiplier_h = multiplier_h // 2
                    setattr(self, f'encode{i}', BasicBlock(1, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    setattr(self, f'p_decode{i}', block(multiplier_v, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    setattr(self, f'c_decode{i}', block(multiplier_v, multiplier_h, kernel_sizes_conv, num_conv=2))
                else:
                    setattr(self, f'encode{i}', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    setattr(self, f'p_decode{i}', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    setattr(self, f'c_decode{i}', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=2))
                if se: setattr(self, f'encode_se{i}', scSE(multiplier_h))
                setattr(self, f'down{i}',
                        nn.Sequential(nn.Conv3d(multiplier_h, multiplier_v, kernel_sizes_down[i], kernel_sizes_transpose[i], padding_down[i]),
                                      nn.InstanceNorm3d(multiplier_v),
                                      nn.SiLU(inplace=True)))
                setattr(self, f'p_deconv{i}', nn.ConvTranspose3d(multiplier_v, multiplier_h, kernel_sizes_down[i], kernel_sizes_conv[i]))
                setattr(self, f'c_deconv{i}', nn.ConvTranspose3d(multiplier_v, multiplier_h, kernel_sizes_down[i], kernel_sizes_conv[i]))
                if se: setattr(self, f'p_decode_se{i}', scSE(multiplier_h))
                if se: setattr(self, f'c_decode_se{i}', scSE(multiplier_h))
                '''if unsupervised:
                    setattr(self, f'u_deconv{i}',
                            nn.ConvTranspose3d(multiplier_v, multiplier_h, kernel_sizes_transpose[i],
                                               kernel_sizes_transpose[i]))
                    setattr(self, f'u_decode{i}',
                            block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    if se: setattr(self, f'u_decode_se{i}', scSE(multiplier_h))'''

            else:
                setattr(self, 'bottleneck', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                if se: setattr(self, f'bottleneck_se', scSE(multiplier_h))

        logit_label_mean = torch.log(label_mean / (1 - label_mean)) * 0.5
        logit_contour_mean = torch.log(contour_mean / (1 - contour_mean)) * 0.5
        if type == 'ResidualBottleneck': base_channels = base_channels // 2
        self.p_out = nn.Conv3d(base_channels, 1, kernel_size=1)
        self.c_out = nn.Conv3d(base_channels, 1, kernel_size=1)
        #with torch.no_grad():
        #    self.p_out.bias.fill_(logit_label_mean)
        #    self.c_out.bias.fill_(logit_contour_mean)
        '''if unsupervised:
            self.u_out = nn.Conv3d(base_channels, 1, kernel_size=1)'''

    def instance_decode(self, bottleneck, encode_features):
        for i in reversed(range(self.depth - 1)):
            if i == self.depth - 2:
                p_x = getattr(self, f"p_deconv{i}")(bottleneck)
                c_x = getattr(self, f"c_deconv{i}")(bottleneck)
            else:
                p_x = getattr(self, f"p_deconv{i}")(p_x)
                c_x = getattr(self, f"c_deconv{i}")(c_x)
            if i != 0:
                p_x += encode_features[i]
                c_x += encode_features[i]
            else:
                p_x = torch.cat((p_x, encode_features[i]), dim=1)
                c_x = torch.cat((c_x, encode_features[i]), dim=1)
            p_x = getattr(self, f"p_decode{i}")(p_x)
            if self.se: p_x = getattr(self, f"p_decode_se{i}")(p_x)
            c_x = getattr(self, f"c_decode{i}")(c_x)
            if self.se: c_x = getattr(self, f"c_decode_se{i}")(c_x)

        p_output, c_output = self.p_out(p_x), self.c_out(c_x)

        return p_output, c_output

    '''def unsupervised_decode(self, bottleneck):
        for i in reversed(range(self.depth - 1)):
            if i == self.depth - 2:
                u_x = getattr(self, f"u_deconv{i}")(bottleneck)
            else:
                u_x = getattr(self, f"u_deconv{i}")(u_x)
            u_x = getattr(self, f"u_decode{i}")(u_x)
            if self.se: u_x = getattr(self, f"u_decode_se{i}")(u_x)

        u_output = self.u_out(u_x)
        return u_output'''

    def forward(self, x):
        encode_features = []

        for i in range(self.depth - 1):
            x = getattr(self, f"encode{i}")(x)
            if self.se: x = getattr(self, f"encode_se{i}")(x)
            encode_features.append(x)
            x = getattr(self, f"down{i}")(x)

        bottleneck = getattr(self, "bottleneck")(x)
        if self.se: bottleneck = getattr(self, f"bottleneck_se")(bottleneck)

        return self.instance_decode(bottleneck, encode_features)

        '''if type[0] == 0:
            return self.instance_decode(bottleneck, encode_features)
        elif type[0] == 1:
            return self.unsupervised_decode(bottleneck)
        elif type[0] == 2:
            return [self.instance_decode(bottleneck, encode_features), self.unsupervised_decode(bottleneck)]
        else:
            raise ValueError("Invalid data type. Should be either '0'(normal) or '1'(unsupervised) or '2'(both).")'''


class UNet(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic',
                 se=False, label_mean=torch.tensor(0.5), contour_mean=torch.tensor(0.5)):
        super(UNet, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.max_pool_flat = nn.MaxPool3d((1, 2, 2))
        self.max_pool_shrink = nn.MaxPool3d((2, 1, 1))
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        self.se = se
        logit_label_mean = torch.log(label_mean / (1 - label_mean)) * 0.5
        logit_contour_mean = torch.log(contour_mean / (1 - contour_mean)) * 0.5
        if abs(self.special_layers) + 1 >= depth:
            raise ValueError("Current Z to XY ratio requires a deeper network.")
        if depth < 2:
            raise ValueError("The depth needs to be at least 2 (2 different feature map size exist).")
        kernel_sizes_conv = (3, 3, 3)

        scale_down = [(1, 2, 2) if self.special_layers > 0 and i < self.special_layers else
                      (2, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                      (2, 2, 2) for i in range(depth)]
        scale_down_kernel_size = [(1, 4, 4) if self.special_layers > 0 and i < self.special_layers else
                                  (4, 1, 1) if self.special_layers < 0 and i < -self.special_layers else
                                  (4, 4, 4) for i in range(depth)]
        padding_down = [(0, 1, 1) if self.special_layers > 0 and i < self.special_layers else
                        (1, 0, 0) if self.special_layers < 0 and i < -self.special_layers else
                        (1, 1, 1) for i in range(depth)]
        block = {'Basic': BasicBlock, 'Residual': ResBasicBlock, 'ResidualBottleneck': ResBottleneckBlock}[type]
        for i in range(depth):
            if i == 0:
                num_conv = 1
            else:
                num_conv = 2
            multiplier_h = base_channels * (2 ** i)
            multiplier_v = base_channels * (2 ** (i+1))
            if i != depth - 1:
                if i == 0:
                    if type == 'ResidualBottleneck':
                        multiplier_h = multiplier_h // 2
                    setattr(self, f'encode{i}', BasicBlock(1, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    setattr(self, f'decode{i}', block(multiplier_v, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    setattr(self, f'p_out{i}', nn.Conv3d(multiplier_h, 1, kernel_size=1))
                    setattr(self, f'c_out{i}', nn.Conv3d(multiplier_h, 1, kernel_size=1))
                else:
                    setattr(self, f'encode{i}', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    setattr(self, f'decode{i}', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    setattr(self, f'p_out{i}', nn.Conv3d(multiplier_h, 1, kernel_size=1))
                    if self.special_layers > 0:
                        depth_factor = max(1, 2**(i - self.special_layers))
                        xy_factor = 2**i
                    else:
                        depth_factor = 2**i
                        xy_factor = max(1, 2**(i - self.special_layers))
                    setattr(self, f'rescale{i}', nn.Upsample(scale_factor=(depth_factor, xy_factor, xy_factor), mode='trilinear', align_corners=True))
                if se: setattr(self, f'encode_se{i}', scSE(multiplier_h))
                setattr(self, f'down{i}', nn.Conv3d(multiplier_h, multiplier_v, scale_down_kernel_size[i], scale_down[i], padding_down[i]))
                setattr(self, f'deconv{i}', nn.ConvTranspose3d(multiplier_v, multiplier_h, scale_down_kernel_size[i], scale_down[i], padding_down[i]))
                if se: setattr(self, f'decode_se{i}', scSE(multiplier_h))
                '''if unsupervised:
                    setattr(self, f'deconv{i}',
                            nn.ConvTranspose3d(multiplier_v, multiplier_h, kernel_sizes_transpose[i],
                                               kernel_sizes_transpose[i]))
                    if se: setattr(self, f'decode_se{i}', scSE(multiplier_h))'''

            else:
                setattr(self, 'bottleneck', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                if se: setattr(self, f'bottleneck_se', scSE(multiplier_h))

        setattr(self, f'c_out', nn.Conv3d(base_channels, 1, kernel_size=1))
        #if unsupervised:
        #    self.u_out = nn.Conv3d(base_channels, 1, kernel_size=1)

    def instance_decode(self, x, encode_features):
        p_outputs = []
        for i in reversed(range(self.depth - 1)):
            x = getattr(self, f"deconv{i}")(x)
            if i != 0:
                x += encode_features[i]
            else:
                x = torch.cat((x, encode_features[i]), dim=1)
            x = getattr(self, f"decode{i}")(x)
            if self.se: x = getattr(self, f"decode_se{i}")(x)
            p = getattr(self, f"p_out{i}")(x)
            if i != 0:
                p = getattr(self, f"rescale{i}")(p)
            p_outputs.append(p)

        c = getattr(self, f"c_out")(x)
        return p_outputs, c

    '''def unsupervised_decode(self, bottleneck):
        for i in reversed(range(self.depth - 1)):
            if i == self.depth - 2:
                u_x = getattr(self, f"u_deconv{i}")(bottleneck)
            else:
                u_x = getattr(self, f"u_deconv{i}")(u_x)
            u_x = getattr(self, f"u_decode{i}")(u_x)
            if self.se: u_x = getattr(self, f"u_decode_se{i}")(u_x)

        u_output = self.u_out(u_x)
        return u_output'''

    def forward(self, x):
        encode_features = []

        for i in range(self.depth - 1):
            x = getattr(self, f"encode{i}")(x)
            if self.se: x = getattr(self, f"encode_se{i}")(x)
            encode_features.append(x)
            x = getattr(self, f"down{i}")(x)

        bottleneck = getattr(self, "bottleneck")(x)
        if self.se: bottleneck = getattr(self, f"bottleneck_se")(bottleneck)

        return self.instance_decode(bottleneck, encode_features)

        '''if type[0] == 0:
            return self.instance_decode(bottleneck, encode_features)
        elif type[0] == 1:
            return self.unsupervised_decode(bottleneck)
        elif type[0] == 2:
            return [self.instance_decode(bottleneck, encode_features), self.unsupervised_decode(bottleneck)]
        else:
            raise ValueError("Invalid data type. Should be either '0'(normal) or '1'(unsupervised) or '2'(both).")'''
