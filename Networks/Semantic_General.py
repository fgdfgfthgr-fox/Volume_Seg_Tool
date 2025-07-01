import torch
import torch.nn as nn
import torch.nn.init as I
import math
from .Modules.General_Components import ResBasicBlock, ResBottleneckBlock, BasicBlock, scSE, AttentionBlock

# Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
# In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich,
# Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

# Çiçek Ö, Abdulkadir A, Lienkamp S S, et al. 3D U-Net: learning dense volumetric segmentation from sparse
# annotation[C]//Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International
# Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19. Springer International Publishing,
# 2016: 424-432.

# This file contains U-net-like Architectures for semantic segmentation,
# with the main difference being the skip connection adds the channels instead of concatenate them
# Basic: Use normal convolution block
# Residual: Use residual block
# ResidualBottleneck: Use residual bottleneck block

# U-net: The most influential DL image classification paper.
# Introduced the concept of skip connection to feed spatial information into the decoder part of the network
# The original UNet implementation in the paper doesn't use batch normalization, nor does conv padding


class UNet(nn.Module):
    def __init__(self, base_channels=64, depth=4, z_to_xy_ratio=1, type='Basic',
                 se=False, label_mean=torch.tensor(0.5)):
        super(UNet, self).__init__()
        self.depth = depth
        self.special_layers = math.floor(math.log2(z_to_xy_ratio))
        self.se = se

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
            multiplier_h = min(base_channels * (2 ** i), 256)
            multiplier_v = min(base_channels * (2 ** (i+1)), 256)
            if i != depth - 1:
                if i == 0:
                    if type == 'ResidualBottleneck':
                        multiplier_h = multiplier_h // 2
                        decode_multiplier_v = multiplier_v // 2
                    else:
                        decode_multiplier_v = multiplier_v
                    setattr(self, f'encode{i}', BasicBlock(1, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    setattr(self, f'decode{i}', BasicBlock(decode_multiplier_v, multiplier_h, kernel_sizes_conv, num_conv=num_conv, norm=False))
                    setattr(self, f'p_out{i}', nn.Conv3d(multiplier_h, 1, kernel_size=1))
                else:
                    setattr(self, f'encode{i}', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv, res_type=res_type))
                    setattr(self, f'decode{i}', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv, norm=False, res_type=res_type))
                    setattr(self, f'p_out{i}', nn.Conv3d(multiplier_h, 1, kernel_size=1))
                    if self.special_layers > 0:
                        depth_factor = max(1, 2**(i - self.special_layers))
                        xy_factor = 2**i
                    else:
                        depth_factor = 2**i
                        xy_factor = max(1, 2**(i - self.special_layers))
                    setattr(self, f'rescale{i}', nn.Upsample(scale_factor=(depth_factor, xy_factor, xy_factor), mode='trilinear', align_corners=False))
                if se: setattr(self, f'encode_se{i}', scSE(multiplier_h))
                if se: setattr(self, f'decode_se{i}', scSE(multiplier_h))
                setattr(self, f'down{i}', nn.Sequential(nn.Conv3d(multiplier_h, multiplier_v, scale_down_kernel_size[i], scale_down[i], padding_down[i]),))
                                                        #nn.InstanceNorm3d(multiplier_v),
                                                        #nn.SiLU(inplace=True)))
                setattr(self, f'deconv{i}', nn.ConvTranspose3d(multiplier_v, multiplier_h, scale_down_kernel_size[i], scale_down[i], padding_down[i]))
                '''if unsupervised:
                    setattr(self, f'u_deconv{i}', nn.ConvTranspose3d(multiplier_v, multiplier_h, kernel_sizes_transpose[i], kernel_sizes_transpose[i]))
                    setattr(self, f'u_decode{i}', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                    if se: setattr(self, f'u_decode_se{i}', scSE(multiplier_h))'''
            else:
                setattr(self, 'bottleneck', block(multiplier_h, multiplier_h, kernel_sizes_conv, num_conv=num_conv))
                if se: setattr(self, f'bottleneck_se', scSE(multiplier_h))
        #logit_label_mean = torch.log(label_mean / (1 - label_mean)) * 0.5 # Multiply by 0.5 seem to make training more stable.
        #if type == 'ResidualBottleneck': base_channels = base_channels // 2
        #with torch.no_grad():
        #    self.s_out.bias.fill_(logit_label_mean)
        '''if unsupervised:
            self.u_out = nn.Conv3d(base_channels, 1, kernel_size=1)'''

    def semantic_decode(self, x, encode_features):
        outputs = []

        for i in reversed(range(self.depth - 1)):

            x = getattr(self, f"deconv{i}")(x)
            if i != 0:
                x += encode_features[i]
            else:
                x = torch.cat((x, encode_features[i]), dim=1)
            x = getattr(self, f"decode{i}")(x)
            if self.se:
                x = getattr(self, f"decode_se{i}")(x)
            output = getattr(self, f"p_out{i}")(x)
            if i != 0:
                output = getattr(self, f"rescale{i}")(output)
            outputs.append(output)

        return outputs

    '''def unsupervised_decode(self, bottleneck):
        for i in reversed(range(self.depth - 1)):
            if i == self.depth - 2:
                u_x = getattr(self, f"u_deconv{i}")(bottleneck)
            else:
                u_x = getattr(self, f"u_deconv{i}")(u_x)
            u_x = getattr(self, f"u_decode{i}")(u_x)
            if self.se:
                u_x = getattr(self, f"u_decode_se{i}")(u_x)
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

        return self.semantic_decode(bottleneck, encode_features)

        '''if type[0] == 0:
            return self.semantic_decode(bottleneck, encode_features)
        elif type[0] == 1:
            return self.unsupervised_decode(bottleneck)
        elif type[0] == 2:
            return [self.semantic_decode(bottleneck, encode_features), self.unsupervised_decode(bottleneck)]
        else:
            raise ValueError("Invalid data type. Should be either '0'(normal) or '1'(unsupervised) or '2'(both).")'''



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # LeakyReLU everywhere → use Kaiming with a = 0.01 (default of LeakyReLU)
        I.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            I.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        if m.out_features == 4:
            # FINAL EMBEDDING LAYER: orthogonal with gain to set Var=0.5
            # orthogonal gives Var=1.0 in each dim → rescale by sqrt(0.5)
            I.orthogonal_(m.weight, gain=math.sqrt(0.5))
            I.zeros_(m.bias)
        else:
            # All the intermediate fcs feeding LeakyReLU
            I.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            I.zeros_(m.bias)