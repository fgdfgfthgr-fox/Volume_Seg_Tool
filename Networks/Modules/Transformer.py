import torch
import torch.nn as nn

class RotaryEmbedding3D(nn.Module):
    def __init__(self, dim, freq=50):
        super().__init__()
        assert dim % 3 == 0, "head dim must be divisible by 3 for 3D rotary embedding"
        self.dim = dim
        self.third_dim = dim // 3

        # Create frequency tensors for depth, height, and width
        inv_freq = 1.0 / (freq ** (torch.arange(0, self.third_dim, 1).float() / self.third_dim))
        self.register_buffer("inv_freq", inv_freq)


    def reshape(self, x):
        return x.reshape(-1, x.shape[-1])

    def forward(self, grid_size):
        D, H, W = grid_size
        device = self.inv_freq.device

        # Create grid coordinates
        d_pos = torch.arange(D, device=device, dtype=self.inv_freq.dtype)
        h_pos = torch.arange(H, device=device, dtype=self.inv_freq.dtype)
        w_pos = torch.arange(W, device=device, dtype=self.inv_freq.dtype)

        # Compute sinusoidal embeddings for depth, height, and width separately
        # Each will have shape: (position, third_dim)
        sin_d = torch.sin(d_pos[:, None] * self.inv_freq[None, :])
        cos_d = torch.cos(d_pos[:, None] * self.inv_freq[None, :])
        sin_h = torch.sin(h_pos[:, None] * self.inv_freq[None, :])
        cos_h = torch.cos(h_pos[:, None] * self.inv_freq[None, :])
        sin_w = torch.sin(w_pos[:, None] * self.inv_freq[None, :])
        cos_w = torch.cos(w_pos[:, None] * self.inv_freq[None, :])

        # Expand to full 3D grid
        # For depth: (D, H, W, third_dim)
        sin_d = sin_d.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)
        cos_d = cos_d.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)
        # For height: (D, H, W, third_dim)
        sin_h = sin_h.unsqueeze(0).unsqueeze(2).repeat(D, 1, W, 1)
        cos_h = cos_h.unsqueeze(0).unsqueeze(2).repeat(D, 1, W, 1)
        # For width: (D, H, W, third_dim)
        sin_w = sin_w.unsqueeze(0).unsqueeze(1).repeat(D, H, 1, 1)
        cos_w = cos_w.unsqueeze(0).unsqueeze(1).repeat(D, H, 1, 1)

        # Flatten spatial dimensions
        sin_d = self.reshape(sin_d)
        cos_d = self.reshape(cos_d)
        sin_h = self.reshape(sin_h)
        cos_h = self.reshape(cos_h)
        sin_w = self.reshape(sin_w)
        cos_w = self.reshape(cos_w)

        return (sin_d, cos_d), (sin_h, cos_h), (sin_w, cos_w)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_3d(x, sin_d, cos_d, sin_h, cos_h, sin_w, cos_w):
    """
    Apply rotary embedding by splitting x into three thirds:
    - First third rotated based on depth (z position)
    - Second third rotated based on height (y position)
    - Last third rotated based on width (x position)
    """
    batch_size, head, seq_len, dim = x.shape
    third_dim = dim // 3

    # Split into three thirds
    x_d = x[..., :third_dim]  # Part to be rotated based on depth
    x_h = x[..., third_dim:2 * third_dim]  # Part to be rotated based on height
    x_w = x[..., 2 * third_dim:]  # Part to be rotated based on width

    # Apply rotary embeddings separately
    x_d_rotated = (x_d * cos_d) + (rotate_half(x_d) * sin_d)
    x_h_rotated = (x_h * cos_h) + (rotate_half(x_h) * sin_h)
    x_w_rotated = (x_w * cos_w) + (rotate_half(x_w) * sin_w)

    # Concatenate back
    return torch.cat([x_d_rotated, x_h_rotated, x_w_rotated], dim=-1)


class MultiheadAttentionWith3DRoPE(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Ensure head_dim is divisible by 3
        assert self.head_dim % 3 == 0, f"head_dim ({self.head_dim}) must be divisible by 3 for 3D RoPE"

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_embedding = RotaryEmbedding3D(self.head_dim)
        #self.reg_embedding = RotaryEmbedding3D(self.head_dim, 4)

    def forward(self, x, grid_size):
        batch_size, seq_len, _ = x.shape
        #reg_idx = seq_len - 8

        #x = self.norm(x)

        # Generate rotary embeddings
        (sin_d, cos_d), (sin_h, cos_h), (sin_w, cos_w) = self.rotary_embedding(grid_size)
        #(sin_dr, cos_dr), (sin_hr, cos_hr), (sin_wr, cos_wr) = self.rotary_embedding((2,2,2))

        # Project and split queries, keys, values
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, nhead, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # unpack along first dimension

        # Apply rotary positional embeddings to queries and keys
        #q = torch.cat((apply_rotary_pos_emb_3d(q[:, :, 0:reg_idx], sin_d, cos_d, sin_h, cos_h, sin_w, cos_w),
        #               apply_rotary_pos_emb_3d(q[:, :, reg_idx:], sin_dr, cos_dr, sin_hr, cos_hr, sin_wr, cos_wr)), dim=2)
        #k = torch.cat((apply_rotary_pos_emb_3d(k[:, :, 0:reg_idx], sin_d, cos_d, sin_h, cos_h, sin_w, cos_w),
        #               apply_rotary_pos_emb_3d(k[:, :, reg_idx:], sin_dr, cos_dr, sin_hr, cos_hr, sin_wr, cos_wr)), dim=2)
        q = apply_rotary_pos_emb_3d(q, sin_d, cos_d, sin_h, cos_h, sin_w, cos_w)
        k = apply_rotary_pos_emb_3d(k, sin_d, cos_d, sin_h, cos_h, sin_w, cos_w)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Compute attention
        attn_output = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )

        # Combine heads and project
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)


# Example usage
if __name__ == "__main__":
    # Parameters
    batch_size = 4
    d_model = 768
    nhead = 16
    grid_size = (8, 16, 16)  # Depth=8, Height=16, Width=16
    seq_len = grid_size[0] * grid_size[1] * grid_size[2]  # 8 * 16 * 16 = 2048

    # Create model
    model = MultiheadAttentionWith3DRoPE(d_model, nhead)

    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = model(x, grid_size)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model head_dim: {model.head_dim}")