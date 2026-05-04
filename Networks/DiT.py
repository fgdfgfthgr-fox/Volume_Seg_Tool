import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .Modules.Swin import SwinTransformerBlock, compute_attn_mask

def unfold(x, kernel_size):
    """
    Unfold a 3D tensor (B, C, D, H, W) using a non‑overlapping 3D sliding window.
    Returns:
        patches: Tensor of shape (B, num_patches, C * kd * kh * kw)
                 where num_patches = (D//kd) * (H//kh) * (W//kw)
    """
    B, C, D, H, W = x.shape
    kd, kh, kw = kernel_size

    x = x.reshape(B, C, D // kd, kd, H // kh, kh, W // kw, kw)

    # Permute to bring block dimensions together: (B, C, Db, Hb, Wb, kd, kh, kw)
    # where Db = D//kd, etc.
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)

    # Flatten kernel dimensions: (B, C, Db*Hb*Wb, kd*kh*kw)
    num_patches = (D // kd) * (H // kh) * (W // kw)
    x = x.reshape(B, C, num_patches, -1)

    # Move the channel dimension to the end: (B, num_patches, C*kd*kh*kw)
    x = x.permute(0, 2, 1, 3).reshape(B, num_patches, -1)

    return x

def fold(patches, kernel_size, original_shape):
    """
    Fold patches back into a 3D tensor (B, C, D, H, W).
    Assumes non‑overlapping windows where kernel size equals stride.

    Args:
        patches: Tensor of shape (B, num_patches, C * kd * kh * kw)
        kernel_size: tuple (kd, kh, kw)
        original_shape: tuple (B, C, D, H, W) of the original tensor

    Returns:
        x: Tensor of shape (B, C, D, H, W)
    """
    B, C, D, H, W = original_shape
    kd, kh, kw = kernel_size

    Db, Hb, Wb = D // kd, H // kh, W // kw  # number of blocks per dimension
    num_patches = Db * Hb * Wb

    # Reshape to separate channels from kernel product
    patches = patches.reshape(B, num_patches, C, kd * kh * kw)
    # Bring channels forward
    patches = patches.permute(0, 2, 1, 3)               # (B, C, num_patches, kd*kh*kw)
    # Split num_patches into block indices and kernel product into spatial dims
    patches = patches.reshape(B, C, Db, Hb, Wb, kd, kh, kw)
    # Interleave block and kernel dimensions (reverse of unfold's permutation)
    patches = patches.permute(0, 1, 2, 5, 3, 6, 4, 7)   # (B, C, Db, kd, Hb, kh, Wb, kw)
    # Merge to full spatial dimensions
    return patches.reshape(B, C, D, H, W)


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
        #self.to_patches = unfoldNd.UnfoldNd(kernel_size=self.patch_size, stride=self.patch_size)
        self.up = nn.Linear(patch_dim, patch_dim*2)
        self.dit = SwinBlock(patch_dim*2, 2, depth, 2, self.window_size)
        self.down_p = nn.Linear(patch_dim*2, patch_dim)
        if instance:
            self.down_c = nn.Linear(patch_dim*2, patch_dim)


    def forward(self, x):

        B,C,D,H,W = x.shape
        grid_size = (D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2])

        # x = self.to_patches(x).permute(0, 2, 1)  # [B, num_patches, d]
        x = unfold(x, self.patch_size)

        x = self.up(x)
        attn_mask = compute_attn_mask(grid_size, self.window_size, self.window_size//2, x.device)
        x = self.dit(x, grid_size, attn_mask)

        p = self.down_p(x)
        # with torch.amp.autocast('cuda', enabled=False):
            # p = unfoldNd.foldNd(p.permute(0, 2, 1), (D, H, W), self.patch_size, stride=self.patch_size)
        p = fold(p, self.patch_size, (B, C, D, H, W))
        if self.instance:
            c = self.down_c(x)
            # with torch.amp.autocast('cuda', enabled=False):
                # c = unfoldNd.foldNd(c.permute(0, 2, 1), (D, H, W), self.patch_size, stride=self.patch_size)
            c = fold(c, self.patch_size, (B, C, D, H, W))
            return p, c
        return p