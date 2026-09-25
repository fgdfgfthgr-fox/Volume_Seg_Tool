import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules.Swin import SwinBlock, compute_attn_mask
from timm.layers import trunc_normal_

def unfold(x, kernel_size):
    """
    x: (B, 1, D, H, W)

    Returns:
        patches: (B, Db, Hb, Wb, kd * kh * kw)
        where Db = D // kd, Hb = H // kh, Wb = W // kw
    """
    B, C, D, H, W = x.shape

    kd, kh, kw = kernel_size
    Db, Hb, Wb = D // kd, H // kh, W // kw

    # (B, Db, kd, Hb, kh, Wb, kw)
    x = x.reshape(B, Db, kd, Hb, kh, Wb, kw)

    # (B, Db, Hb, Wb, kd, kh, kw)
    x = x.permute(0, 1, 3, 5, 2, 4, 6)

    # (B, Db, Hb, Wb, kd*kh*kw)
    return x.reshape(B, Db, Hb, Wb, kd * kh * kw)

def fold(patches, kernel_size, original_shape):
    """
    patches: (B, Db, Hb, Wb, kd * kh * kw)
    original_shape: (B, D, H, W) or (B, 1, D, H, W)

    Returns:
        x: (B, 1, D, H, W)
    """
    B, C, D, H, W = original_shape

    kd, kh, kw = kernel_size
    Db, Hb, Wb = D // kd, H // kh, W // kw

    # (B, Db, Hb, Wb, kd, kh, kw)
    x = patches.reshape(B, Db, Hb, Wb, kd, kh, kw)

    # (B, Db, kd, Hb, kh, Wb, kw)
    x = x.permute(0, 1, 4, 2, 5, 3, 6)

    # (B, D, H, W)
    x = x.reshape(B, D, H, W)

    x = x.unsqueeze(1)
    return x


class SwinTransformer(nn.Module):
    def __init__(self, patch_size=(2,4,4), window_size=4, depth=8, instance=True):
        super().__init__()

        self.patch_size = patch_size
        self.window_size = window_size
        self.instance = instance
        self.patch_dim = self.patch_size[0]*self.patch_size[1]*self.patch_size[2]
        self.model_dim = 16 * ((self.patch_dim * 3) // 16)
        self.up = nn.Linear(self.patch_dim, self.model_dim)
        #self.patchify = nn.Conv3d(1, self.model_dim, self.patch_size, self.patch_size)
        self.dit = SwinBlock(self.model_dim, 2, depth, 2, self.window_size)
        self.down_p = nn.Linear(self.model_dim, self.patch_dim)
        #self.down_p = nn.ConvTranspose3d(self.model_dim, 1, self.patch_size, self.patch_size)
        if instance:
            self.down_c = nn.Linear(self.model_dim, self.patch_dim)
            #self.down_c = nn.ConvTranspose3d(self.model_dim, 1, self.patch_size, self.patch_size)

    def reshape_back(self, x, B, d, h, w, D, H, W):
        x = x.permute(0, 4, 1, 2, 3).reshape(B, 1, *self.patch_size, d, h, w)  # (B, 1, p, p, p, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(B, 1, D, H, W)
        return x

    def forward(self, x):
        B,C,D,H,W = x.shape
        # grid_size = (D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2])

        x = unfold(x, self.patch_size)  # [B, D/p, H/p, W/p, dim]
        #x = self.patchify(x) # (B, dim, D/p, H/p, W/p)
        d, h, w = x.shape[1:4]

        x = self.up(x)
        #x = x + self.abs
        attn_mask = compute_attn_mask((d, h, w), self.window_size, self.window_size//2, x.device).to(x.dtype) # (num_windows, 1, N, N)
        x = self.dit(x, attn_mask)

        p = self.down_p(x)
        #p = self.reshape_back(p, B, d, h, w, D, H, W)
        p = fold(p, self.patch_size, (B, C, D, H, W))
        if self.instance:
            c = self.down_c(x)
            c = fold(c, self.patch_size, (B, C, D, H, W))
            #c = self.reshape_back(c, B, d, h, w, D, H, W)
            return p, c
        return p


class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, *kwargs):
        super().__init__(*kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.down_s = nn.Linear(self.model_dim, self.patch_dim)
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, x, mask):
        B,C,D,H,W = x.shape
        grid_size = (D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2])

        x = unfold(x, self.patch_size)

        x = self.up(x)
        attn_mask = compute_attn_mask(grid_size, self.window_size, self.window_size // 2, x.device)

        B, L, _ = x.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        x = self.dit(x, grid_size, attn_mask)

        x = self.down_s(x)
        x = fold(x, self.patch_size, (B, C, D, H, W))
        return x

def norm_targets_3d(targets, patch_size):
    """
    Normalise each voxel by the mean and std of a local 3D patch.
    patch_size: int (odd) or tuple of three odd ints (pd, ph, pw).
    """
    if isinstance(patch_size, int):
        assert patch_size % 2 == 1
        patch_size = (patch_size, patch_size, patch_size)
    else:
        assert all(p % 2 == 1 for p in patch_size)

    pd, ph, pw = patch_size

    # Average pooling over local neighbourhood
    mean = F.avg_pool3d(targets, kernel_size=patch_size, stride=1,
                        padding=(pd//2, ph//2, pw//2), count_include_pad=False)

    # Variance using E[x^2] - E[x]^2, with unbiased correction
    square_mean = F.avg_pool3d(targets ** 2, kernel_size=patch_size, stride=1,
                               padding=(pd//2, ph//2, pw//2), count_include_pad=False)
    # Count of valid (non‑padded) elements in each window
    ones = torch.ones_like(targets)
    count = F.avg_pool3d(ones, kernel_size=patch_size, stride=1,
                         padding=(pd//2, ph//2, pw//2), count_include_pad=True) * (pd * ph * pw)

    var = (square_mean - mean ** 2) * (count / (count - 1).clamp(min=1))
    var = torch.clamp(var, min=0.0)

    normed = (targets - mean) / (var + 1e-6).sqrt()
    return normed

class SimMIM(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.patch_size = encoder.patch_size

    def forward(self, x, mask):
        x_rec = self.encoder(x, mask)

        pd, ph, pw = self.patch_size
        B, C, D, H, W = x.shape

        # Reshape the 1D mask to the 3D grid of patches
        mask_3d = mask.view(B, D // pd, H // ph, W // pw)

        # Upsample by repeating each patch element along the corresponding spatial dimensions
        mask_expanded = mask_3d.repeat_interleave(pd, dim=1) \
            .repeat_interleave(ph, dim=2) \
            .repeat_interleave(pw, dim=3)

        # Add a singleton channel dimension
        mask_expanded = mask_expanded.unsqueeze(1)  # (B, 1, D, H, W)

        #with torch.no_grad():  # normalisation is a fixed preprocessing
        #    x = norm_targets_3d(x, 25)

        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask_expanded).sum() / (mask_expanded.sum() + 1e-5)
        return loss, x_rec