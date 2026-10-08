import torch
import torch.nn as nn
import math


class LocalDecoderMLP(nn.Module):
    def __init__(self, patch_size, in_channels, cond_hidden_size):
        super().__init__()
        patch_dim = patch_size[0]*patch_size[1]*patch_size[2]*in_channels
        self.first = nn.Linear(patch_dim, 2 * patch_dim)
        self.act = nn.SiLU()
        self.second = nn.Linear(2*patch_dim + cond_hidden_size, 2*patch_dim)
        self.third = nn.Linear(2*patch_dim, patch_dim)

    def forward(self, x, c):
        # x: (B, L, patch_dim), c: (B, L, cond_hidden_size)
        x = self.act(self.first(x))
        x = self.act(self.second(torch.cat([x, c], dim=2)))
        x = self.third(x)
        return x


def _unet_schedule(patch_size):
    """Compute U-Net depth and per-level downsampling pattern from patch size.

    patch_size: (D, H, W) with H == W, all powers of 2.
    Returns (depth, patterns) where patterns is a list of length depth, each entry
    in {'d', 'sym', 'hw'}:
      'd'   -> downsample only the depth axis
      'sym' -> downsample depth and H/W
      'hw'  -> downsample only H/W
    """
    pd, ph, pw = patch_size
    assert ph == pw, "H and W patch sizes must be equal"
    ld = int(round(math.log2(pd)))
    lh = int(round(math.log2(ph)))
    depth = max(ld, lh)
    assert depth >= 1, "patch size must have at least one dimension >= 2"

    if ld >= lh:
        patterns = ['d'] * (ld - lh) + ['sym'] * lh
    else:
        patterns = ['sym'] * ld + ['hw'] * (lh - ld)
    return depth, patterns


def _new_down_params(pattern):
    """Down (strided conv) params for LocalDecoderNew."""
    if pattern == 'd':
        return dict(kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
    elif pattern == 'sym':
        return dict(kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
    else:  # 'hw'
        return dict(kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))


def _new_up_params(pattern):
    """Up (transposed conv) params for LocalDecoderNew (kernel scale 4)."""
    if pattern == 'd':
        return dict(kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
    elif pattern == 'sym':
        return dict(kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
    else:  # 'hw'
        return dict(kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))


def _old_pool_params(pattern):
    """MaxPool params for LocalDecoder (kernel 2 equivalent)."""
    if pattern == 'd':
        return dict(kernel_size=(2, 1, 1), stride=(2, 1, 1))
    elif pattern == 'sym':
        return dict(kernel_size=(2, 2, 2), stride=(2, 2, 2))
    else:  # 'hw'
        return dict(kernel_size=(1, 2, 2), stride=(1, 2, 2))


def _old_up_params(pattern):
    """Transposed conv params for LocalDecoder (kernel scale 2)."""
    if pattern == 'd':
        return dict(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
    elif pattern == 'sym':
        return dict(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
    else:  # 'hw'
        return dict(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))


class LocalDecoderUNetNew(nn.Module):
    """Per-patch U-Net decoder: (patch, global condition) -> denoised patch.

    Strided-conv variant. Depth is inferred from patch_size so the bottleneck
    is spatially 1x1x1.
    """

    def __init__(self, patch_size, in_channels, cond_hidden_size, base_channels=16):
        super().__init__()

        depth, patterns = _unet_schedule(patch_size)
        chs = [base_channels * (2 ** i) for i in range(depth)]

        self.input = nn.Sequential(
            nn.Conv3d(in_channels, chs[0], kernel_size=3, padding=1, padding_mode='replicate'),
            nn.SiLU(),
        )

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for ch, pattern in zip(chs, patterns):
            self.enc.append(nn.Sequential(
                nn.Conv3d(ch, ch, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.SiLU(),
            ))
            self.down.append(nn.Conv3d(ch, ch * 2, **_new_down_params(pattern)))

        # Bottleneck: inject global condition
        self.bottleneck = nn.Sequential(
            nn.Conv3d(chs[-1] * 2 + cond_hidden_size, chs[-1] * 2, kernel_size=1),
            nn.SiLU(),
        )

        # Decoder upsampling + fusion blocks (reverse of encoder order)
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i, pattern in reversed(list(enumerate(patterns))):
            self.ups.append(nn.ConvTranspose3d(chs[i] * 2, chs[i], **_new_up_params(pattern)))
            self.decs.append(nn.Sequential(
                nn.Conv3d(chs[i] * 2, chs[i], kernel_size=3, padding=1, padding_mode='replicate'),
                nn.SiLU(),
            ))

        self.out_conv = nn.Conv3d(chs[0], in_channels, kernel_size=1)

    def forward(self, x, c):
        # x: [B, C, D, H, W], c: [B, cond_hidden_size, 1, 1, 1]
        skips = []
        x = self.input(x)

        for enc, down in zip(self.enc, self.down):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(torch.cat([x, c], dim=1))

        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))

        return self.out_conv(x)


class LocalDecoderUNet(nn.Module):
    """Per-patch U-Net decoder: (patch, global condition) -> denoised patch.

    Max-pool variant. Depth is inferred from patch_size so the bottleneck
    is spatially 1x1x1.
    """

    def __init__(self, patch_size, in_channels, cond_hidden_size, base_channels=16):
        super().__init__()

        depth, patterns = _unet_schedule(patch_size)
        chs = [base_channels * (2 ** i) for i in range(depth)]

        # Encoder blocks (channel-changing convs)
        self.enc = nn.ModuleList()
        prev_ch = in_channels
        for ch in chs:
            self.enc.append(nn.Sequential(
                nn.Conv3d(prev_ch, ch, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.SiLU(),
            ))
            prev_ch = ch

        # Per-level pooling (symmetric / depth-only / H-W-only)
        self.pool = nn.ModuleList()
        for pattern in patterns:
            self.pool.append(nn.MaxPool3d(**_old_pool_params(pattern)))

        # Bottleneck: inject global condition
        self.bottleneck = nn.Sequential(
            nn.Conv3d(chs[-1] + cond_hidden_size, chs[-1], kernel_size=1),
            nn.SiLU(),
        )

        # Decoder upsampling + fusion blocks
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i, pattern in reversed(list(enumerate(patterns))):
            self.ups.append(nn.ConvTranspose3d(chs[i], chs[i], **_old_up_params(pattern)))
            out_ch = chs[i - 1] if i > 0 else chs[0]
            self.decs.append(nn.Sequential(
                nn.Conv3d(chs[i] * 2, out_ch, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.SiLU(),
            ))

        self.out_conv = nn.Conv3d(chs[0], in_channels, kernel_size=1)

    def forward(self, x, c):
        # x: [B, C, D, H, W], c: [B, cond_hidden_size, 1, 1, 1]
        skips = []

        for enc, pool in zip(self.enc, self.pool):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(torch.cat([x, c], dim=1))

        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))

        return self.out_conv(x)