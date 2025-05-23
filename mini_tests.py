import torch
import threading
import subprocess
import lightning.pytorch as pl
from Components import DataComponents
from Networks.GANS import Generator, Discriminator
from pytorch_lightning.loggers import TensorBoardLogger
from pl_module import PLModule


def start_tensorboard():
    subprocess.run("tensorboard --logdir='lightning_logs'", shell=True)


if __name__ == "__main__":
    generator = Generator(n_mlp=3, style_dim=128, final_resolution=(128, 128, 128), lr_mlp=0.01, base_channels=8, depth=4, z_to_xy_ratio=1)
    output = generator(4)
    Discriminator = Discriminator((128, 128, 128), 8, 4, 1)
    output = Discriminator(output)
    print(output.size())