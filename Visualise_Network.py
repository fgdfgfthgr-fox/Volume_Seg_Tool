import math

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn as nn
from Networks import *
from Components.DataComponents import path_to_tensor
from Networks.Modules.EMNets_Components import TLU
import time

# The filename for the existing network to read.
EXISTING_NETWORK_NAME = "placeholder.pth"

# The architecture of the network
NETWORK_ARCH = Iterations_New.Initial(base_channels=8)

# The image file you are going to exam the network with.
INPUT = 'MockDatasets/train/testimg_2.tif'


class V_N_PLModule(pl.LightningModule):

    def __init__(self, network_arch):
        super(V_N_PLModule, self).__init__()
        self.model = network_arch
        self.activations = []
        self.sigmoids = []

        # Register a hook for activations layers
        def activation_hook_fn(module, input, output):
            self.activations.append(output)

        def sigmoid_hook_fn(module, input, output):
            self.sigmoids.append(output)

        # Register the hook for all (P)ReLU layers in the model
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(activation_hook_fn)
            if isinstance(module, nn.PReLU):
                module.register_forward_hook(activation_hook_fn)
            if isinstance(module, TLU):
                module.register_forward_hook(activation_hook_fn)
            if isinstance(module, nn.Sigmoid):
                module.register_forward_hook(sigmoid_hook_fn)
            if isinstance(module, nn.GELU):
                module.register_forward_hook(sigmoid_hook_fn)

    def forward(self, image):
        return self.model(image)


if __name__ == "__main__":
    model = V_N_PLModule(NETWORK_ARCH)
    model.load_state_dict(torch.load(EXISTING_NETWORK_NAME))
    test_tensor = path_to_tensor(INPUT).unsqueeze(0).unsqueeze(0)  # Shape of (1, 1, 128, 256, 256)
    # Set the model to evaluation mode
    # model.eval()

    # Pass the test tensor through the model
    with torch.no_grad():
        model(test_tensor)

    plt.figure()
    plt.suptitle('Input')
    input = test_tensor[:, :, 0:1, :, :].squeeze()
    plt.imshow(input, cmap='gist_gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    # Now, self.activations contains the intermediate activations from (P)ReLU layers
    for i, activation in enumerate(model.activations):
        if i <= 60:
            pass
        else:
            plt.figure(figsize=(16,9))
            if len(activation.shape) == 5:
                activation = activation[:, :, 0:1, :, :]
                tensor_width = activation.shape[-1]
                tensor_height = activation.shape[-2]
                if tensor_width >= 4 and tensor_height >= 4:
                    channels = torch.split(activation, 1, dim=1)
                    num_channels = len(channels)
                    plt.suptitle(f'Activation Layer {i}, {num_channels} channels, {tensor_width} * {tensor_height}')

                    # Create a grid of subplots to display channels
                    rows = math.floor(math.sqrt(num_channels))
                    cols = math.ceil(num_channels/rows)

                    for j, channel in enumerate(channels):
                        channel = channel.squeeze()
                        plt.subplot(rows, cols, j + 1)
                        plt.imshow(channel.cpu().numpy(), cmap='gist_gray', interpolation='nearest')
                        plt.axis('off')

                    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce spacing between subplots
                    plt.show()
                    time.sleep(0.5)

    plt.figure()
    plt.suptitle(f'Sigmoid Layer')
    sigmoid = model.sigmoids[-1][:, :, 0:1, :, :]
    channels = torch.split(sigmoid, 1, dim=1)
    num_channels = len(channels)

    rows = math.floor(math.sqrt(num_channels))
    cols = math.ceil(num_channels/rows)

    for j, channel in enumerate(channels):
        channel = channel.squeeze()
        plt.subplot(rows, cols, j + 1)
        plt.imshow(channel.cpu().numpy(), cmap='gist_gray', interpolation='nearest')
        plt.colorbar()
        plt.axis('off')
    plt.show()

    plt.figure()
    plt.suptitle(f'Output')
    output = torch.where(sigmoid >= 0.5, 1, 0).to(torch.int16)
    plt.imshow(output.squeeze().cpu().numpy(), cmap='gist_gray', interpolation='nearest')
    plt.axis('off')
    plt.show()