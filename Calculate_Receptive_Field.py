import torch
import torch.nn as nn
from Networks import *


class MaxReceptiveFieldCalculator(nn.Module):
    def __init__(self):
        super(MaxReceptiveFieldCalculator, self).__init__()

    def forward(self, model):
        receptive_field = 1
        current_stride = 1

        def hook_fn(module, input, output):
            nonlocal receptive_field, current_stride
            layer = module
            if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.ConvTranspose3d):
                dilation = layer.dilation[0] if hasattr(layer, 'dilation') else 1
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                receptive_field += (kernel_size - 1) * current_stride
                current_stride *= stride
                if dilation > 1:
                    receptive_field *= dilation

        hooks = []
        for layer in model.modules():
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

        # Forward pass through the network to calculate receptive field
        input_tensor = torch.randn(1, 1, 64, 64, 64)  # Adjust input size as needed
        model(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return receptive_field


# Create an instance of the network
net = UNets.UNet(base_channels=16, depth=3)

# Create an instance of the receptive field calculator
rf_calculator = MaxReceptiveFieldCalculator()

# Calculate the maximum receptive field
max_receptive_field = rf_calculator(net)

print("Maximum Receptive Field:", max_receptive_field)