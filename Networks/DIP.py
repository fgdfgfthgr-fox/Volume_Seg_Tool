import math
import torch
import torch.nn as nn
from .Modules.Swin import compute_attn_mask, SwinBlock
from .Modules.local_decoders import LocalDecoderMLP
from .DiT import fold, unfold


class Network(nn.Module):
    def __init__(self, patch_size=(2,4,4), window_size=4, depth=8, instance=True):
        super().__init__()
        self.nhead = 2
        self.patch_size = patch_size
        self.window_size = window_size
        self.instance = instance

        self.patch_dim = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        self.model_dim = (8*self.nhead) * ((self.patch_dim * 2) // (8*self.nhead))

        self.up = nn.Linear(self.patch_dim, self.model_dim)
        self.dit = SwinBlock(self.model_dim, self.nhead, depth, 2, self.window_size)
        self.local_decoder_p = LocalDecoderMLP(
            patch_size=patch_size,
            in_channels=1,
            cond_hidden_size=self.model_dim,
        )
        if instance:
            self.local_decoder_c = LocalDecoderMLP(
                patch_size=patch_size,
                in_channels=1,
                cond_hidden_size=self.model_dim,
            )

    def forward(self, x):
        B, _, D, H, W = x.shape

        x = unfold(x, self.patch_size)  # [B, D/p, H/p, W/p, dim]
        d, h, w = x.shape[1:4]
        attn_mask = compute_attn_mask((d, h, w), self.window_size, self.window_size // 2, x.device).to(x.dtype)  # (num_windows, 1, N, N)
        s = self.up(x)

        s = self.dit(s, attn_mask)
        s = s.reshape(B, d*h*w, self.model_dim)

        x_raw_local = x.reshape(B, d*h*w, self.patch_dim)
        out_p = self.local_decoder_p(x_raw_local, s)
        out_p = fold(out_p, self.patch_size, (B, _, D, H, W))

        if self.instance:
            out_c = self.local_decoder_c(x_raw_local, s)
            out_c = fold(out_c, self.patch_size, (B, _, D, H, W))
            return out_p, out_c
        return out_p