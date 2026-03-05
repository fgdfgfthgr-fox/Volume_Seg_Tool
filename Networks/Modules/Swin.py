import torch
import torch.nn as nn
from timm.layers import trunc_normal_

# Mostly implementation from Timm, just changed to 3D

def to_3tuple(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 3
        return tuple(x)
    return (x, x, x)


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window depth, height, width

    Returns:
        windows: (num_windows*B, wd, wh, ww, C)
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
    ).reshape(-1, wd, wh, ww, C)
    return windows


def window_reverse(windows, window_size, D, H, W):
    """
    Args:
        windows: (num_windows*B, wd, wh, ww, C)
        window_size (tuple[int]): window depth, height, width
        D, H, W: original depth, height, width

    Returns:
        x: (B, D, H, W, C)
    """
    wd, wh, ww = window_size
    # Calculate batch size
    windows_per_dim = (D // wd, H // wh, W // ww)
    num_windows_per_image = windows_per_dim[0] * windows_per_dim[1] * windows_per_dim[2]
    B = windows.shape[0] // num_windows_per_image

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
    Supports both shifted and non-shifted windows.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Depth, height, width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wd, wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) where N = wd*wh*ww
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, head_dim

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, num_heads, N, N

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)  # N, N, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, N, N
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply mask if provided (for SW-MSA)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def compute_attn_mask(input_resolution, window_size, shift_size, device):
    """
    Compute the attention mask for shifted window attention (SW-MSA).

    Args:
        input_resolution (tuple[int]): (D, H, W) of the input feature map.
        window_size (int | tuple[int]): (wd, wh, ww) – window dimensions.
        shift_size (int | tuple[int]): (sd, sh, sw) – shift amounts in each dimension.
        device: The device the final mask is on.

    Returns:
        attn_mask: mask with shape (num_windows, N, N) where N = wd*wh*ww,
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
    d_idx = torch.arange(D)
    h_idx = torch.arange(H)
    w_idx = torch.arange(W)

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
    attn_mask = torch.where(attn_mask != 0, -100.0, 0.0)
    return attn_mask.to(device)


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
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=2., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = to_3tuple(window_size)
        self.shift_size = to_3tuple(shift_size) if isinstance(shift_size, (tuple, list)) else (shift_size, shift_size, shift_size)
        self.mlp_ratio = mlp_ratio

        # Ensure shift_size is in [0, window_size)
        for i in range(3):
            assert 0 <= self.shift_size[i] < self.window_size[i], "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim, elementwise_affine=False)

        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.norm2 = norm_layer(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(mlp_ratio * dim), dim),
        )


    def forward(self, x, input_resolution, attn_mask=None):
        """
        x: (B, L, C) where L = D*H*W
        """
        D, H, W = input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, wd, wh, ww, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size**3, C

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)  # B, D, H, W, C

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x

        x = x.view(B, D * H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))

        return x