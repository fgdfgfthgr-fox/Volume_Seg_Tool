import torch

def generate_simmim_mask(tensor, mask_patch_size, model_patch_size, mask_ratio):
    """
    Generate binary masks for SimMIM masking on a batch of 3D volumes.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        mask_patch_size (tuple): Size of the coarse mask patches (md, mh, mw).
        model_patch_size (tuple): Size of the model patches (kd, kh, kw) used for tokenisation.
        mask_ratio (tuple): Range (min_ratio, max_ratio) from which the actual mask ratio is sampled.

    Returns:
        torch.Tensor: Binary mask of shape (B, num_patches) with 1 indicating masked tokens.
    """
    B, C, D, H, W = tensor.shape
    kd, kh, kw = model_patch_size
    md, mh, mw = mask_patch_size
    r_min, r_max = mask_ratio

    # Sample the mask ratio once for the whole batch
    r = torch.empty(1, device=tensor.device).uniform_(r_min, r_max).item()

    # Number of model patches in each dimension
    num_patches_d = D // kd
    num_patches_h = H // kh
    num_patches_w = W // kw
    num_patches = num_patches_d * num_patches_h * num_patches_w

    # Size of the coarse mask grid (ceil division to cover the entire volume)
    grid_d = (D + md - 1) // md
    grid_h = (H + mh - 1) // mh
    grid_w = (W + mw - 1) // mw

    # Compute coarse grid index for each model patch (same for all batch elements)
    patch_i = torch.arange(num_patches_d, device=tensor.device)
    patch_j = torch.arange(num_patches_h, device=tensor.device)
    patch_k = torch.arange(num_patches_w, device=tensor.device)

    # Starting coordinate of each patch
    start_i = patch_i * kd
    start_j = patch_j * kh
    start_k = patch_k * kw

    # Coarse cell index for each patch
    coarse_i = start_i // md
    coarse_j = start_j // mh
    coarse_k = start_k // mw

    # Create a 3D grid of coarse indices and flatten to 1D index into the coarse mask
    i_grid, j_grid, k_grid = torch.meshgrid(coarse_i, coarse_j, coarse_k, indexing='ij')
    flat_coarse_idx = i_grid * (grid_h * grid_w) + j_grid * grid_w + k_grid
    flat_coarse_idx = flat_coarse_idx.flatten()  # shape: (num_patches,)

    # Generate coarse mask for all batch elements simultaneously
    # shape: (B, grid_d, grid_h, grid_w)
    coarse_mask_all = torch.rand(B, grid_d, grid_h, grid_w, device=tensor.device) < r
    coarse_mask_all = coarse_mask_all.float()  # convert to 0/1

    # Flatten spatial dimensions of coarse mask
    coarse_mask_flat = coarse_mask_all.view(B, -1)  # shape: (B, grid_d*grid_h*grid_w)

    # Gather the mask values for each patch index
    mask = coarse_mask_flat[:, flat_coarse_idx]  # shape: (B, num_patches)

    return mask

if __name__ == '__main__':
    test_tensor = torch.randn(1, 1, 16, 16, 16)
    mask = generate_simmim_mask(test_tensor, mask_patch_size=(4,4,4), model_patch_size=(2,2,2), mask_ratio=(0.5,0.55))
    print(mask.shape)