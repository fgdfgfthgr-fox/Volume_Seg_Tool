import torch.nn as nn
import unfoldNd
from torch.utils.checkpoint import checkpoint

from .Modules.Swin import SwinTransformerBlock, compute_attn_mask

class SwinBlock(nn.Module):
    def __init__(self, d_main, nhead, num_layers, ffn_multiplier, window_size):
        super().__init__()
        self.num_layers = num_layers
        self.layers_no_shift = nn.ModuleList([
            SwinTransformerBlock(d_main, nhead, window_size, 0, ffn_multiplier, False, nn.RMSNorm)
            for i in range(num_layers)
        ])
        self.layers_shift = nn.ModuleList([
            SwinTransformerBlock(d_main, nhead, window_size, window_size // 2, ffn_multiplier, False, nn.RMSNorm)
            for i in range(num_layers)
        ])

    def forward(self, x, input_resolution, attn_mask):
        for layer_n, layer_s in zip(self.layers_no_shift, self.layers_shift):
            x = layer_n(x, input_resolution, None) if not self.training else checkpoint(layer_n, x, input_resolution, None, use_reentrant=False)
            x = layer_s(x, input_resolution, attn_mask) if not self.training else checkpoint(layer_s, x, input_resolution, attn_mask, use_reentrant=False)
        return x


class Network(nn.Module):
    def __init__(self, patch_size=(2,4,4), window_size=4, depth=8, instance=True):
        super().__init__()

        self.patch_size = patch_size
        self.window_size = window_size
        self.instance = instance
        patch_dim = (self.patch_size[0]*self.patch_size[1]*self.patch_size[2])
        self.to_patches = unfoldNd.UnfoldNd(kernel_size=self.patch_size, stride=self.patch_size)
        self.up = nn.Linear(patch_dim, patch_dim*2)
        self.dit = SwinBlock(patch_dim*2, 2, depth, 2, self.window_size)
        self.down_p = nn.Linear(patch_dim*2, patch_dim)
        if instance:
            self.down_c = nn.Linear(patch_dim*2, patch_dim)


    def forward(self, x):

        B,C,D,H,W = x.shape
        grid_size = (D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2])

        x = self.to_patches(x).permute(0, 2, 1)  # [B, num_patches, d]

        x = self.up(x)
        attn_mask = compute_attn_mask(grid_size, self.window_size, self.window_size//2, x.device)
        x = self.dit(x, grid_size, attn_mask)

        p = self.down_p(x)
        p = unfoldNd.foldNd(p.permute(0, 2, 1), (D, H, W), self.patch_size, stride=self.patch_size)
        if self.instance:
            c = self.down_c(x)
            c = unfoldNd.foldNd(c.permute(0, 2, 1), (D, H, W), self.patch_size, stride=self.patch_size)
            return p, c
        return p