import math

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn as nn
from Networks import *
from Components.DataComponents import path_to_tensor
from pl_module import pick_arch
import time



class V_N_PLModule(pl.LightningModule):

    def __init__(self, arch_args):
        super(V_N_PLModule, self).__init__()
        self.network = pick_arch(arch_args)
        self.activations = []
        self.c_outs = []
        self.p_outs = []
        self.unsupervised = False

        # Register a hook for activations layers
        def activation_hook_fn(module, input, output):
            self.activations.append(output)

        def c_outs_hook_fn(module, input, output):
            self.c_outs.append(torch.sigmoid(output))

        def p_outs_hook_fn(module, input, output):
            self.p_outs.append(torch.sigmoid(output))

        # Register the hook for all relevant layers in the model
        for name, module in self.network.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(activation_hook_fn)
            if isinstance(module, nn.PReLU):
                module.register_forward_hook(activation_hook_fn)
            if isinstance(module, nn.GELU):
                module.register_forward_hook(activation_hook_fn)
            if isinstance(module, nn.CELU):
                module.register_forward_hook(activation_hook_fn)
            if isinstance(module, nn.SiLU):
                module.register_forward_hook(activation_hook_fn)
            '''if name == "u_out":
                self.unsupervised = True
                module.register_forward_hook(u_outs_hook_fn)'''
            if name == "c_out":
                module.register_forward_hook(c_outs_hook_fn)
            if "rescale" in name or name == "p_out0":
                module.register_forward_hook(p_outs_hook_fn)

    def forward(self, image):
        '''if self.unsupervised:
            return self.model(image, [2,])
        else:
            return self.model(image, [0,])'''
        return self.network(image)



if __name__ == "__main__":

    # The filename for the existing network to read.
    EXISTING_NETWORK_NAME = "''/example_name.ckpt"

    # The architecture of the network
    NETWORK_ARCH = Semantic_General.UNet(16, 4, 1, 'Residual', True, 0.5)

    # The image file you are going to exam the network with.
    INPUT = 'Datasets/mid_visualiser/image.tif'

    model = V_N_PLModule.load_from_checkpoint(EXISTING_NETWORK_NAME).to('cpu')
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

    # Now, self.activations contains the intermediate activations
    for i, activation in enumerate(model.activations):
        if i <= 50:
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
    sigmoid = model.s_outs[-1][:, :, 0:1, :, :].squeeze()
    plt.imshow(sigmoid.cpu().numpy(), cmap='gist_gray', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.suptitle(f'Unsupervised Output Layer')
    output = model.u_outs[-1][:, :, 0:1, :, :].squeeze()
    plt.imshow(output.cpu().numpy(), cmap='gist_gray', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.show()
