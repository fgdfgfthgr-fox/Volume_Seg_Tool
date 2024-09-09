import torch
import math

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_sinusoidal_encoding(num_positions, embedding_dim):
    """Generates sinusoidal positional encodings."""
    position = torch.arange(num_positions, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / embedding_dim))
    pos_encoding = torch.zeros((num_positions, embedding_dim), device=device)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding

def image_sequentialization(x, patch_size=(2,2,2)):
    """
    Args:
        x (torch.Tensor): Input tensor of shape (Batch, Channel, Depth, Height, Width).
        patch_size (tuple): Size of each patch (depth, height, width). Default: (2,2,2)

    Returns:
        torch.Tensor: Sequentialized patches with sinusoidal positional encoding added.
    """
    B, C, D, H, W = x.shape
    patch_D, patch_H, patch_W = patch_size

    # Calculate number of patches in each dimension
    num_patches_D = D // patch_D
    num_patches_H = H // patch_H
    num_patches_W = W // patch_W
    num_patches = num_patches_D * num_patches_H * num_patches_W

    # Extract patches and flatten them
    patches = x.unfold(2, patch_D, patch_D).unfold(3, patch_H, patch_H).unfold(4, patch_W, patch_W)
    patches = patches.contiguous().view(B, C, num_patches, -1)

    # Rearrange to (Batch, Num_Patches, Patch_Volume*Channel)
    patches = patches.permute(0, 2, 1, 3).contiguous().view(B, num_patches, -1)

    # Generate and add sinusoidal positional encodings
    embed_dim = patches.shape[-1]
    pos_encoding = generate_sinusoidal_encoding(num_patches, embed_dim).unsqueeze(0)
    patches += pos_encoding

    return patches


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels):
        """
        Args:
            patch_size (tuple): Size of each patch (depth, height, width).
            in_channels (int): Number of input channels.
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.embed_dim = patch_volume * in_channels

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channel, Depth, Height, Width).

        Returns:
            torch.Tensor: Patch embeddings of shape (Batch, Num_Patches, Embed_Dim).
        """
        patches = image_sequentialization(x, self.patch_size)
        return patches


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        ff_dim = embed_dim * 3
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.feedforward(x)
        x = x + ff_output
        x = self.norm2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, patch_size, in_channels, num_heads, num_layers):
        super(Transformer, self).__init__()
        patch_volume = patch_size ** 3
        embed_dim = patch_volume * in_channels
        #projection_dim = embed_dim // patch_size
        self.patch_embedding = PatchEmbedding((patch_size, patch_size, patch_size), in_channels)
        #self.projection = nn.Linear(embed_dim, projection_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        #self.reprojection = nn.Linear(embed_dim, embed_dim*2)

    def forward(self, x):
        embedded_patches = self.patch_embedding(x)
        #projected_patches = self.projection(embedded_patches)
        for block in self.transformer_blocks:
            embedded_patches = block(embedded_patches)
        #embedded_patches = self.reprojection(projected_patches)
        # Reshape to original 3D volume
        reshaped_output = reshape_to_original(embedded_patches, x.shape, self.patch_embedding.patch_size)

        return reshaped_output


def reshape_to_original(x, original_shape, patch_size):
    B, C, D, H, W = original_shape
    patch_D, patch_H, patch_W = patch_size
    num_patches_D = D // patch_D
    num_patches_H = H // patch_H
    num_patches_W = W // patch_W
    x = x.view(B, num_patches_D, num_patches_H, num_patches_W, C, patch_D, patch_H, patch_W)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    x = x.view(B, C, D, H, W)
    return x
