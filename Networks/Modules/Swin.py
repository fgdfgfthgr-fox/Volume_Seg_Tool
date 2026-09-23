import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint
from aule import scaled_dot_product_attention

# Mostly implementation from Timm, just changed to 3D

def to_3tuple(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 3
        return tuple(x)
    return (x, x, x)

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1)*alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window depth, height, width

    Returns:
        windows: (B*num_windows, wd, wh, ww, C)
    """
    B, D, H, W, C = x.shape
    wd, wh, ww = window_size
    # Reshape into grid of windows
    x = x.view(
        B,  # Batch
        D // wd, wd,  # Depth windows, window depth
        H // wh, wh,  # Height windows, window height
        W // ww, ww,  # Width windows, window width
        C  # Channels
    )

    # Rearrange dimensions to bring windows together
    windows = x.permute(
        0,  # B
        1, 3, 5,  # D_windows, H_windows, W_windows
        2, 4, 6,  # wd, wh, ww (window interior)
        7  # C
    ).reshape(B, -1, wd, wh, ww, C)
    return windows


def window_reverse(windows, window_size, D, H, W):
    """
    Args:
        windows: (B, num_windows, wd, wh, ww, C)
        window_size (tuple[int]): window depth, height, width
        D, H, W: original depth, height, width

    Returns:
        x: (B, D, H, W, C)
    """
    wd, wh, ww = window_size
    # Calculate batch size
    windows_per_dim = (D // wd, H // wh, W // ww)
    B = windows.shape[0]

    # Reshape windows back to grid
    x = windows.view(
        B,  # Batch
        windows_per_dim[0],  # D_windows
        windows_per_dim[1],  # H_windows
        windows_per_dim[2],  # W_windows
        wd, wh, ww,  # Window interior
        -1  # Channels
    )

    # Rearrange and reshape to original dimensions
    x = x.permute(
        0,  # B
        1, 4,  # D_windows, wd
        2, 5,  # H_windows, wh
        3, 6,  # W_windows, ww
        7  # C
    ).reshape(B, D, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    3D Window based multi-head self attention with relative position bias.

    Args:
        dim (int): Number of input channels. Also the dim of the value.
        window_size (tuple[int]): Depth, height, width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: False
        qk_dim (int, optional): Dimension of query, key (before divide by num_heads). Default: None
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_dim=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wd, wh, ww)
        self.num_heads = num_heads
        self.qk_dim = qk_dim if qk_dim is not None else dim
        assert self.qk_dim % num_heads == 0, "qk_dim must be divisible by num_heads"
        self.head_dim_qk = self.qk_dim // num_heads

        # Relative position bias table: shape (2*wd-1)*(2*wh-1)*(2*ww-1), nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

        # Compute pair-wise relative position index for each token inside the 3D window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # 3, wd, wh, ww
        coords_flatten = torch.flatten(coords, 1)  # 3, wd*wh*ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 3
        # Shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        # Multiply to get a unique index: for depth, multiply by (2*wh-1)*(2*ww-1); for height, multiply by (2*ww-1)
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, self.qk_dim * 2 + dim, bias=qkv_bias)
        self.q_norm = DyT(self.head_dim_qk)
        self.k_norm = DyT(self.head_dim_qk)
        self.proj = nn.Linear(dim, dim)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, Np, N, C = x.shape # [B, num_windows, wd*wh*ww, C]

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.dim], dim=-1)
        q = q.reshape(B, Np, N, self.num_heads, self.head_dim_qk).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, Np, N, self.num_heads, self.head_dim_qk).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, Np, N, self.num_heads, self.dim//self.num_heads).permute(0, 1, 3, 2, 4)
        # each: (B, num_windows, num_heads, N, head_dim)

        q = self.q_norm(q).to(x.dtype)
        k = self.k_norm(k).to(x.dtype)

        # Relative position bias: (num_heads, N, N)
        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0).to(x.dtype)  # (num_heads, N, N)

        if mask is None:
            out = scaled_dot_product_attention(q, k, v, attn_mask=rel_bias)
        else:
            # Combine window mask and relative position bias
            # mask:      (nW, 1, N, N)
            # rel_bias:  (1, num_heads, N, N)
            # combined:  (B, nW, num_heads, N, N)
            rel_bias = mask + rel_bias  # (nW, num_heads, N, N)
            rel_bias = rel_bias.unsqueeze(0)  # (1, nW, num_heads, N, N)

            out = scaled_dot_product_attention(q, k, v, attn_mask=rel_bias)

        # Reshape and project (same as original)
        out = out.transpose(2, 3).reshape(B, Np, N, C)
        out = self.proj(out)
        return out


def compute_attn_mask(input_resolution, window_size, shift_size, device):
    """
    Compute the attention mask for shifted window attention (SW-MSA).

    Args:
        input_resolution (tuple[int]): (D, H, W) of the input feature map.
        window_size (int | tuple[int]): (wd, wh, ww) – window dimensions.
        shift_size (int | tuple[int]): (sd, sh, sw) – shift amounts in each dimension.
        device: The device the final mask is on.

    Returns:
        attn_mask: mask with shape (num_windows, 1, N, N) where N = wd*wh*ww,
                   or None if no shift is applied.
    """
    D, H, W = input_resolution
    window_size = to_3tuple(window_size)
    shift_size = to_3tuple(shift_size)
    wd, wh, ww = window_size
    sd, sh, sw = shift_size

    if sd == 0 and sh == 0 and sw == 0:
        return None

    # Coordinate tensors
    d_idx = torch.arange(D, device=device)
    h_idx = torch.arange(H, device=device)
    w_idx = torch.arange(W, device=device)

    # Segment labels: 0 (lower), 1 (middle), 2 (upper)
    d_label = (d_idx >= (D - wd)).int() + (d_idx >= (D - sd)).int()
    h_label = (h_idx >= (H - wh)).int() + (h_idx >= (H - sh)).int()
    w_label = (w_idx >= (W - ww)).int() + (w_idx >= (W - sw)).int()

    # Combine labels into unique region IDs (shape: D, H, W)
    ids = d_label[:, None, None] * 9 + h_label[None, :, None] * 3 + w_label[None, None, :]
    img_mask = ids[None, ..., None]  # add batch and channel dims → (1, D, H, W, 1)

    # Proceed with the standard window partitioning and mask creation
    mask_windows = window_partition(img_mask, window_size)          # (num_windows, wd, wh, ww, 1)
    mask_windows = mask_windows.view(-1, wd * wh * ww)              # (num_windows, N)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (num_windows, N, N)
    attn_mask = torch.where(attn_mask != 0, -torch.inf, 0.0)
    return attn_mask.unsqueeze(1)


class SwinTransformerBlock(nn.Module):
    """
    3D Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int | tuple[int]): Window size (depth, height, width).
        shift_size (int | tuple[int]): Shift size for SW-MSA (same for all dims if int, else per dim).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: False
        qk_dim (int, optional): Dimension of query, key (before divide by num_heads). Default: None
        more_ffn (bool, optional): If True, perform another mlp after the first mlp block. Default: False
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=2., qkv_bias=False, qk_dim=None, more_ffn=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = to_3tuple(window_size)
        self.shift_size = to_3tuple(shift_size) if isinstance(shift_size, (tuple, list)) else (shift_size, shift_size, shift_size)
        self.mlp_ratio = mlp_ratio
        self.more_ffn = more_ffn

        # Ensure shift_size is in [0, window_size)
        for i in range(3):
            assert 0 <= self.shift_size[i] < self.window_size[i], "shift_size must be in 0-window_size"

        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_dim=qk_dim)
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones((dim)),requires_grad=True)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(mlp_ratio * dim), dim),
        )
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones((dim)),requires_grad=True)
        if more_ffn:
            self.mlp2 = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(mlp_ratio * dim), dim),
        )
            self.gamma_3 = nn.Parameter(1e-4 * torch.ones((dim)),requires_grad=True)

    def full_attn(self, x, attn_mask):
        B, D, H, W, C = x.shape #  [B, D/p, H/p, W/p, dim]

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                              dims=(1, 2, 3))

        x = window_partition(x, self.window_size)  # B, nW, wd, wh, ww, C
        x = x.view(B, -1, self.window_size[0] * self.window_size[1] * self.window_size[2], C) # [B, num_windows, wd*wh*ww, C]

        # W-MSA / SW-MSA
        x = self.attn(x, mask=attn_mask)

        x = x.view(B, -1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        x = window_reverse(x, self.window_size, D, H, W)  # B, D, H, W, C

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                              dims=(1, 2, 3))

        return x

    def forward(self, x, attn_mask=None):
        """
        x: (B, L, C) where L = D*H*W
        """
        x = x + self.gamma_1.to(x.dtype) * self.full_attn(x, attn_mask)

        # FFN
        x = x + self.gamma_2.to(x.dtype) * self.mlp(x)
        if self.more_ffn:
            x = x + self.gamma_3.to(x.dtype) * self.mlp2(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, d_main, nhead, num_layers, ffn_multiplier, window_size):
        super().__init__()
        self.num_layers = num_layers
        slope = np.linspace(0.5, 0.25, num_layers)
        d_qk = [int((8*nhead) * (slope[i] * d_main // (8*nhead))) for i in range(num_layers)]
        self.layers_no_shift = nn.ModuleList([
            SwinTransformerBlock(d_main, nhead, window_size, 0, ffn_multiplier, False, d_qk[i], True)
            for i in range(num_layers)
        ])
        self.layers_shift = nn.ModuleList([
            SwinTransformerBlock(d_main, nhead, window_size, window_size // 2, ffn_multiplier, False, d_qk[i])
            for i in range(num_layers)
        ])

    def forward(self, x, attn_mask):
        for layer_n, layer_s in zip(self.layers_no_shift, self.layers_shift):
            x = layer_n(x, None) if not self.training else checkpoint(layer_n, x, None, use_reentrant=False)
            x = layer_s(x, attn_mask) if not self.training else checkpoint(layer_s, x, attn_mask, use_reentrant=False)
        return x