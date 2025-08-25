import torch.nn.functional as F
import torchvision.transforms.v2.functional as T_F
import torchvision.transforms.v2 as transforms
import random
import torch
import math
import scipy
import os
import numpy as np
from joblib import Parallel, delayed

from scipy.ndimage import distance_transform_edt
import torch.multiprocessing as mp
from .Perlin3d import generate_perlin_noise_3d


device = "cuda" if torch.cuda.is_available() else "cpu"
# Various customize image augmentation implementations specialised in 4 dimensional tensors.
# (Channel, Depth, Height, Width).


def sim_low_res(tensor, scale=2):
    """
    Simulate low resolution by down-sampling using nearest-neighbor interpolation and then up-sampling using cubic/linear
    interpolation.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W) or (C, H, W).
        scale (float): Scale factor for down-sampling and up-sampling. Default is 2.

    Returns:
        torch.Tensor: Simulated low-resolution tensor with the same shape as the input.
    """
    shape = tensor.shape
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, scale_factor=scale, mode='nearest-exact')
    if len(shape) == 4:
        tensor = F.interpolate(tensor, size=[shape[1], shape[2], shape[3]], mode='trilinear')
    if len(shape) == 3:
        tensor = F.interpolate(tensor, size=[shape[1], shape[2]], mode='bicubic')
    return tensor.squeeze(0)


def adj_gamma(tensor, gamma, gain=1):
    """
    Adjust gamma correction for a tensor of shape (C, D, H, W).

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W).
        gamma (float): Non-negative gamma correction factor.
        gain (float, optional): Multiplicative gain. Default is 1.

    Returns:
        torch.Tensor: Gamma-adjusted 3D tensor with the same shape as the input.
    """
    min, max = tensor.min(), tensor.max()
    if min - max == 0:
        return tensor  # Avoid division by zero; return as-is
    tensor = (tensor-min)/(max-min)
    tensor = (tensor ** gamma) * gain
    tensor = (tensor * (max-min)) + min
    return tensor


def adj_contrast(tensor, contrast):
    """
    Adjust contrast for a tensor of shape (C, D, H, W).

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W).
        contrast (float): Contrast adjustment factor.

    Returns:
        torch.Tensor: Contrast-adjusted 3D tensor with the same shape as the input.
    """
    tensor *= contrast
    return tensor


def adj_brightness(tensor, brightness):
    """
    Adjust brightness for a tensor of shape (C, D, H, W).

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W).
        brightness (float): Brightness adjustment factor.

    Returns:
        torch.Tensor: Brightness-adjusted 3D tensor with the same shape as the input.
    """
    return torch.clamp(brightness + tensor, -4, 4)


def gaussian_blur_3d(tensor, kernel_size=3, sigma=0.8):
    """
    Apply 3D Gaussian blur to a 3D tensor.
    High kernel_size could slow down augmentation significantly.

    Args:
        tensor (torch.Tensor): Input 3D tensor of shape {C, D, H, W}.
        kernel_size (int): Size of the Gaussian kernel. Default: 3.
        sigma (float): Sigma of the Gaussian blur. Default: 0.8.

    Returns:
        torch.Tensor: 3D tensor after applying Gaussian blur with the same shape as the input.
    """
    # Generate 1D Gaussian kernel along each dimension
    input_channels = tensor.shape[0]
    #sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    x = torch.arange(-math.floor(kernel_size/2), math.ceil(kernel_size/2), dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)

    # Extend the 1D kernel to 3D
    kernel = torch.Tensor(kernel_1d[:,None,None]*kernel_1d[None,:,None]*kernel_1d[None,None,:])
    kernel = kernel / kernel.sum()

    # Extend the kernel to the 5D filter required for torch.nn.functional.conv3d
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(input_channels, input_channels, 1, 1, 1)

    # Apply 3D convolution
    blurred_output = torch.nn.functional.conv3d(tensor.unsqueeze(0), kernel, padding=kernel_size // 2).squeeze(0)

    return blurred_output


def rotate_4d_tensor(tensor, planes=['xy', 'xz', 'yz'], angles=[0., 0., 0.],
                     interpolation='bilinear'):
    """
    Rotate a 4D PyTorch tensor on the xy, xz, and yz planes simultaneously.

    Args:
        tensor (torch.Tensor): Tensor of shape (Channel, Depth, Height, Width).
        planes (List of str): the planes to rotate ('xy', 'xz', 'yz').
        angles (List of floats): the rotation angles for each plane.
        interpolation (str): the interpolation method ('nearest' or 'bilinear').

    Returns:
        torch.Tensor
    """

    _, D, H, W = tensor.shape

    # Create a base affine matrix for each rotation plane
    affine_matrices = []

    for plane, ang in zip(planes, angles):
        theta = math.radians(ang)

        if plane == 'xy':
            # Rotation in the xy plane
            rotation_matrix = torch.tensor([
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta),  math.cos(theta), 0],
                [0, 0, 1]
            ])
        elif plane == 'xz':
            # Rotation in the xz plane
            rotation_matrix = torch.tensor([
                [math.cos(theta), 0, -math.sin(theta)],
                [0, 1, 0],
                [math.sin(theta), 0,  math.cos(theta)]
            ])
        elif plane == 'yz':
            # Rotation in the yz plane
            rotation_matrix = torch.tensor([
                [1, 0, 0],
                [0, math.cos(theta), -math.sin(theta)],
                [0, math.sin(theta),  math.cos(theta)]
            ])
        affine_matrices.append(rotation_matrix)

    # Combine all rotation matrices
    total_rotation = affine_matrices[0]
    for matrix in affine_matrices[1:]:
        total_rotation = total_rotation @ matrix

    # Create the affine matrix in 4D (4x4 matrix)
    affine_4x4 = torch.eye(4)
    affine_4x4[:3, :3] = total_rotation

    # Create the affine grid
    grid = F.affine_grid(affine_4x4[:3, :].unsqueeze(0), tensor.unsqueeze(0).shape, align_corners=False)

    # Apply the grid sample using the specified interpolation
    rotated = F.grid_sample(tensor.to(torch.float32).unsqueeze(0), grid, mode=interpolation, padding_mode='reflection', align_corners=False)

    return rotated.squeeze(0)


def custom_rand_crop_rotate(tensors, depth, height, width,
                            angle_range=((0, 360), (0, 360), (0, 360)),
                            planes=['xy'], interpolations=('bilinear', 'nearest'),  # fill_values=(0, 0),
                            ensure_bothground=True, max_attempts=50, minimal_foreground=0.01, minimal_background=0.02,
                            zarr=False, img_mean=0, img_std=1):
    """
    Randomly crop then rotate a list of 3D PyTorch tensors given the desired depth, height, and width, preferably with at least some foreground object.\n
    Whether it contains foreground object is determined by the second tensor in the list,
    pixels with values larger or equal to 1 are considered foreground.\n
    If no foreground object is found after max_attempts attempts, it will output a warning message and crop a random volume.

    Args:
        tensors (list of torch.Tensor or torch.Tensor): List of input tensors of shape (Channel, Depth, Height, Width).
        depth (int): Desired depth of the cropped tensors.
        height (int): Desired height of the cropped tensors.
        width (int): Desired width of the cropped tensors.
        angle_range (list of tuple): Range of rotation angles (min_angle, max_angle) for each plane in degrees.
        planes (list or None): The plane(s) on which the random rotation will be applied: ['xy', 'xz', 'yz']. Default: ['xy']
        interpolations (tuple): Interpolation methods for each tensor ('nearest' or 'bilinear').
        ensure_bothground (bool): If True, will try to ensure that the output contains both foreground and background objects according to the settings below. (default: True)
        max_attempts (int): Maximum number of attempts to find a crop with at least 1% foreground object (default: 50).
        minimal_foreground (float): Proportion of desired minimal foreground pixels (default: 0.01).
        minimal_background (float): Proportion of desired minimal background pixels (default: 0.02).
        zarr (bool): Needs to be set to true if the input are not torch tensor but zarr arrays. (default: False)

    Returns:
        List of cropped tensors or just the cropped tensor itself.
    """

    def contains_bothground(tensor):
        total_pixels = tensor.numel()
        pixels_greater_than_zero = (tensor > 0).sum().detach()

        return (total_pixels * minimal_foreground) <= pixels_greater_than_zero <= (total_pixels * (1 - minimal_background))

    # Suppose all the tensors in the list are the exact same shape.
    c, d, h, w = tensors[0].shape

    if d < depth or h < height or w < width:
        # Calculate the amount of padding needed for each dimension
        d_pad = max(0, depth - d)
        h_pad = max(0, height - h)
        w_pad = max(0, width - w)
        padded = []
        for tensor in tensors:
            # Pad the tensor if needed
            tensor = torch.nn.functional.pad(tensor, (0, w_pad, 0, h_pad, 0, d_pad))
            padded.append(tensor)
        c, d, h, w = padded[0].shape
    else:
        padded = tensors

    def rotation(tensors):
        if len(planes) == 0:
            return tensors
        angles = [random.uniform(*angle) for angle in angle_range]
        rotated_tensors = []
        for idx, tensor in enumerate(tensors):
            rotated_tensor = tensor.clone()
            rotated_tensor = rotate_4d_tensor(rotated_tensor, planes, angles, interpolations[idx])
            rotated_tensors.append(rotated_tensor)

        return rotated_tensors

    def cropping(tensors):
        # Randomly select crop starting locations within the valid range
        d_offset = random.randint(0, d - depth)
        h_offset = random.randint(0, h - height)
        w_offset = random.randint(0, w - width)
        cropped_tensors = []
        for tensor in tensors:
            cropped_tensor = tensor[:, d_offset:d_offset + depth,
                                       h_offset:h_offset + height,
                                       w_offset:w_offset + width]
            if zarr:
                cropped_tensor = torch.from_numpy(cropped_tensor)
            cropped_tensors.append(cropped_tensor.to(torch.float32, copy=False))
        if zarr:
            cropped_tensors[0] = (cropped_tensors[0] - img_mean) / img_std
        return cropped_tensors

    if ensure_bothground:
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            cropped_tensors = cropping(padded)
            if len(planes) != 0:
                rotated_tensors = rotation(cropped_tensors)
            else:
                rotated_tensors = cropped_tensors

            # Check if the label tensor (2nd tensor) contains a foreground object
            if contains_bothground(rotated_tensors[1]):
                return rotated_tensors
        # If no suitable crop is found after max_attempts, raise a warning
        print(f"Random clop failed: No suitable crop with desired threshold found after {max_attempts} attempts. Will "
              f"return a random crop.")
        return cropping(padded)
    else:
        cropped_tensors = cropping(padded)
        rotated_tensors = rotation(cropped_tensors)
        return rotated_tensors


def random_gradient(tensor, e_range=(0.5, 1.5), mode='gamma'):
    """
    Apply random gamma or contrast or brightness adjustment to a PyTorch tensor, with a gradient pattern.

    Args:
        tensor (torch.Tensor): Input image tensor of shape [channel, depth, height, width].
        range (tuple): Range for random contrast adjustment.
        mode (str): 'gamma' or 'contrast' or 'brightness'.

    Returns:
        torch.Tensor: Adjusted image tensor.
    """

    a, b = random.uniform(e_range[0], e_range[1]), random.uniform(e_range[0], e_range[1])
    low, high = min(a, b), max(a, b)

    # Generate a random side (left, right, top, bottom, front, back) for the gradient effect
    gradient_side = torch.randint(0, 6, size=(1,))

    # Generate a gradient tensor for gamma adjustment based on the selected side
    if gradient_side == 0:  # Left side
        gradient = torch.linspace(low, high, tensor.shape[3])
        gradient = gradient.view(1, 1, 1, -1)
    elif gradient_side == 1:  # Right side
        gradient = torch.linspace(low, high, tensor.shape[3])
        gradient = gradient.view(1, 1, 1, -1)
        gradient = gradient.flip(dims=(3,))
    elif gradient_side == 2:  # Top side
        gradient = torch.linspace(low, high, tensor.shape[2])
        gradient = gradient.view(1, 1, -1, 1)
    elif gradient_side == 3:  # Bottom side
        gradient = torch.linspace(low, high, tensor.shape[2])
        gradient = gradient.view(1, 1, -1, 1)
        gradient = gradient.flip(dims=(2,))
    elif gradient_side == 4:  # Front side
        gradient = torch.linspace(low, high, tensor.shape[1])
        gradient = gradient.view(1, -1, 1, 1)
    else:  # Back side
        gradient = torch.linspace(low, high, tensor.shape[1])
        gradient = gradient.view(1, -1, 1, 1)
        gradient = gradient.flip(dims=(1,))
    if mode == 'gamma':
        tensor_min, tensor_max = tensor.min(), tensor.max()
        if tensor_max - tensor_min == 0:
            return tensor  # Avoid division by zero; return as-is
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        tensor = (tensor ** gradient)
        tensor = (tensor * (tensor_max - tensor_min)) + tensor_min
        return tensor
    elif mode == 'contrast':
        return torch.clamp(gradient * tensor, -4, 4)
    elif mode == 'brightness':
        return torch.clamp(gradient + tensor, -4, 4)


def salt_and_pepper_noise(tensor, prob=0.01):
    """
    Add salt and pepper noise to a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor to which noise will be added.
        prob (float): Probability of adding 'salt' (maximum) and 'pepper' (minimum) noise to each element.

    Returns:
        torch.Tensor: Tensor with salt and pepper noise added.
    """

    # Add salt noise
    salt_mask = torch.rand_like(tensor, device=device) < prob
    tensor[salt_mask] = tensor.max()

    # Add pepper noise
    pepper_mask = torch.rand_like(tensor, device=device) < prob
    tensor[pepper_mask] = tensor.min()

    return tensor


def exclude_border_labels(array, inward, outward):
    """
    Exclude the borders of the objects in the label tensor. Also transform it to sparsely annotated form.

    Args:
        array (np.Array): Input tensor of shape (D,H,W). 0 for background, 1 for foreground
        inward (int): Size of morphological erosion.
        outward (int):  Size of morphological dilation.

    Returns:
        torch.Tensor: transformed Tensor. 0 for unlabeled (excluded label), 1 for foreground, 2 for background
    """

    structuring_element = np.ones((3, 3, 3), dtype=np.int8)
    # Erosion
    if inward >= 1:
        eroded_array = scipy.ndimage.binary_erosion(array, structuring_element, inward).astype(np.int8)
    else:
        eroded_array = array
    # Dilation
    if outward >= 1:
        dilated_array = scipy.ndimage.binary_dilation(array, structuring_element, outward).astype(np.int8)
    else:
        dilated_array = array
    eroded_tensor, dilated_tensor = torch.from_numpy(eroded_array), torch.from_numpy(dilated_array)

    # Identify excluded regions and set their values to 0
    excluded_regions = (eroded_tensor == 0) & (dilated_tensor == 1)

    transformed_array = np.full_like(array, 2)
    transformed_array[array == 1] = 1  # Set foreground to 1
    transformed_array[excluded_regions] = 0  # Set excluded regions to 0
    return transformed_array


def binarisation(tensor):
    """
    A quick and dirty way to convert an instance labelled tensor to a semantic labelled tensor.
    """
    #tensor = np.clip(tensor, 0, 1).astype(np.bool_)
    tensor = np.where(tensor>=1, np.bool_(True), np.bool_(False))
    return tensor


def binary_dilation_torch(input, structure=None, iterations=1, mask=None,
                          output=None):
    """
    Multidimensional binary dilation with the given structuring element.
    Similar to scipy.ndimage.binary_dilation but is for Pytorch Tensors.

    Parameters
    ----------
    input : torch.Tensor
        Binary tensor to be dilated. Non-zero (True) elements form
        the subset to be dilated.
    structure : torch.Tensor, optional
        Structuring element used for the dilation. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one.
    iterations : int, optional
        The dilation is repeated `iterations` times (one, by default).
        If iterations is less than 1, the dilation is repeated until the
        result does not change anymore. Only an integer of iterations is
        accepted.
    mask : torch.Tensor, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.
    output : torch.Tensor, optional
        Tensor of the same shape as input, into which the output is placed.
        By default, a new tensor is created.

    Returns
    -------
    binary_dilation : torch.Tensor of bools
        Dilation of the input by the structuring element.
    """

    if structure is None:
        structure = {
            1: torch.ones(3, dtype=input.dtype, device=input.device),
            2: torch.ones((3, 3), dtype=input.dtype, device=input.device),
            3: torch.ones((3, 3, 3), dtype=input.dtype, device=input.device)
        }[input.dim()]
    unsqueezed_structure = structure.unsqueeze(0).unsqueeze(0).float()

    # Determine the appropriate convolution function based on the dimensionality of the input
    conv_function = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[input.dim()]

    # Perform binary dilation using convolution
    for _ in range(iterations):
        unsqueezed_input = input.unsqueeze(0).unsqueeze(0).float()
        dilated_input = conv_function(unsqueezed_input, unsqueezed_structure, padding=1)
        dilated_input = dilated_input.squeeze().bool()

        # Apply mask if provided
        if mask is not None:
            dilated_input[mask == 0] = input[mask == 0]

        input = dilated_input

    # Return the result
    return input if output is None else input.copy_(output)


def instance_contour_transform(input_array, contour_inward=0, contour_outward=1):
    """
    Transform an instance segmented map into a contour map.\n
    The contour is generated using morphological erosion and dilation.

    Args:
        input_array (np.Array): The input instance segmented map with a shape of (D, H, W).
                                     0 is background and every other value is a distinct object.
        contour_inward (int): Size of morphological erosion. No longer used.
        contour_outward (int): Size of morphological dilation.

    Returns:
        torch.Tensor: Contour map where 0 are background or inside of objects while 1 are the boundaries.
    """
    input_array = input_array
    unique_values = np.unique(input_array)
    transformed_tensor = np.zeros_like(input_array, dtype=np.bool_)
    structure = np.ones((3, 3, 3), dtype=transformed_tensor.dtype)
    def process_object(value):
        if value == 0:
            return  # Skip background

        # Create a binary mask for the current object
        object_mask = (input_array == value)

        # Calculate bounding box for the object
        non_zero_indices = np.nonzero(object_mask)
        min_indices = np.min(non_zero_indices[0]), np.min(non_zero_indices[1]), np.min(non_zero_indices[2])
        max_indices = np.max(non_zero_indices[0]), np.max(non_zero_indices[1]), np.max(non_zero_indices[2])

        # Add padding for dilation
        pad = contour_outward
        min_indices = np.clip(np.array([min_indices[0] - pad, min_indices[1] - pad, min_indices[2] - pad]), 0, None)
        max_indices = np.clip(np.array([max_indices[0] + pad+1, max_indices[1] + pad+1, max_indices[2] + pad+1]),
                              None,
                              np.array([input_array.shape[0], input_array.shape[1], input_array.shape[2]]))

        # Crop the map according to the bounding box
        cropped_object = object_mask[min_indices[0]:max_indices[0],
                                     min_indices[1]:max_indices[1],
                                     min_indices[2]:max_indices[2]]

        # Perform dilation on the cropped object
        dilated_object = scipy.ndimage.binary_dilation(cropped_object, iterations=contour_outward, structure=structure)

        # Mark boundary area as one
        excluded_regions = (cropped_object == 0) & (dilated_object == 1)

        transformed_tensor[min_indices[0]:max_indices[0],
                           min_indices[1]:max_indices[1],
                           min_indices[2]:max_indices[2]] += excluded_regions

    # Use joblib to parallelize the loop
    Parallel(backend='threading', n_jobs=-1)(delayed(process_object)(value) for value in unique_values)
    return transformed_tensor


def nearest_interpolate(input_tensor, target_size):
    """
    Scaling a tensor into target size.\n
    The native torch.nn.functional.interpolate doesn't work on tensors with dtype of uint16 or uint32.
    So I have to implement my own.

    Args:
        input_tensor (torch.Tensor): The input tensor with a shape of (B, C, D, H, W).
        target_size (tuple or list): Output spacial size. In (D, H, W).
    """
    batch, channels, depth, height, width = input_tensor.shape
    new_depth, new_height, new_width = target_size

    depth_indices = torch.linspace(0, depth-1, new_depth, device=input_tensor.device).round().long()
    height_indices = torch.linspace(0, height-1, new_height, device=input_tensor.device).round().long()
    width_indices = torch.linspace(0, width-1, new_width, device=input_tensor.device).round().long()

    output_tensor = input_tensor[:, :,
                                 depth_indices[:, None, None],
                                 height_indices[None, :, None],
                                 width_indices[None, None, :]]

    return output_tensor


def edge_replicate_pad(input_tensors, padding_percentile=0.1):
    """
    Crops the input tensors to a smaller version, and replicate pad the lost region.

    Args:
        input_tensors (tuple of torch.Tensor): Input tensors with a shape of (C, D, H, W)
        padding_percentile (float): The maximal percentile of cropping. Default is 0.1

    Return:
        output_tensor (list of torch.Tensor): Output tensors.
    """
    crop_percentile = random.uniform(0, padding_percentile)
    C, D, H, W = input_tensors[0].shape
    D_crop = max(int(D * crop_percentile), 1)
    H_crop = max(int(H * crop_percentile), 1)
    W_crop = max(int(W * crop_percentile), 1)
    output_tensors = []
    for input_tensor in input_tensors:
        output_tensor = input_tensor[:, D_crop:-D_crop, H_crop:-H_crop, W_crop:-W_crop]
        output_tensor = F.pad(output_tensor, [W_crop, W_crop, H_crop, H_crop, D_crop, D_crop], mode='replicate')
        output_tensors.append(output_tensor)
    return output_tensors


def middle_z_normalize(input_tensor, z_percentile=0.75):
    """
    Normalise image to be between 0 and 1. The lowest 1%
    However, only the middle chunk along the Z dimension of 3D microscopy images will be used for calculating the mean and
    standard deviation of image intensity.

    Args:
        input_tensor (torch.Tensor): Input tensor with a shape of (D, H, W)
        z_percentile (float): The central percentile of the image which will be used for calculating the mean and standard deviation of image intensity.

    Return:
        output_tensor (torch.Tensor)
    """
    z, x, y = input_tensor.shape
    low_z, high_z = int((0.5-(z_percentile/2))*z), max(int((0.5+(z_percentile/2))*z), 1)
    #middle_chunk = input_tensor[low_z:high_z, :, :]
    low, high = np.percentile(input_tensor[low_z:high_z, :, :], 1), np.percentile(input_tensor[low_z:high_z, :, :], 99)
    #print(low, high)
    input_tensor = (input_tensor - low) / (high - low)
    input_tensor = torch.clamp(input_tensor, 0, 1)
    return input_tensor


def gaussian_noise(input_tensor, strength=0.05, octaves=3):
    """
    Add gaussian noise of different resolutions to the tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor with a shape of (C, D, H, W)
        strength (float): The intensity of the noise.
        octaves (int): Will generate this many additional low-res noises and interpolate them to target shape before combining.

    Return:
        output_tensor (torch.Tensor)
    """
    origin_shape = input_tensor.shape
    noises = torch.zeros_like(input_tensor, dtype=torch.float32)
    for octave in range(0, octaves):
        # Since we are only dealing with images of single channel, don't worry the channel dim got downscaled as well.
        shape = [max(int(shape / 2**octave), 1) for shape in origin_shape]
        noise = (torch.randn(shape, dtype=torch.float32) * random.uniform(0.5, 1.5) * (0.5**octave) * strength).unsqueeze(0)
        noises += F.interpolate(noise, origin_shape[1:], mode='trilinear', align_corners=False).squeeze(0)
    input_tensor += noises
    input_tensor = torch.clamp(input_tensor, -4, 4)
    return input_tensor


def perlin_noise(input_tensor, strength, max_res):
    shapes = input_tensor.shape
    xy_to_z_ratio = shapes[1]/shapes[0]
    res = random.randint(2, max_res)
    ress = ((max(int(res/xy_to_z_ratio), 2)), res, res)
    noise = generate_perlin_noise_3d([shape for shape in shapes], ress)
    noise *= strength
    input_tensor += noise
    input_tensor = torch.clamp(input_tensor, -4, 4)
    return input_tensor


def remove_black_borders(volumes):
    """
    Remove black borders.
    """
    non_list = False
    if not isinstance(volumes, list):
        volumes = (volumes, )
        non_list = True
    # Sum along the height and width dimensions to find non-black slices along depth
    non_black_depth = np.any(volumes[0], axis=(1, 2))
    non_black_height = np.any(volumes[0], axis=(0, 2))
    non_black_width = np.any(volumes[0], axis=(0, 1))

    # Find the first and last indices where the image is not black
    depth_start, depth_end = non_black_depth.nonzero()[0][[0, -1]]
    height_start, height_end = non_black_height.nonzero()[0][[0, -1]]
    width_start, width_end = non_black_width.nonzero()[0][[0, -1]]

    # Crop the volumes using the computed indices
    volumes = [volume[depth_start:depth_end + 1,
                      height_start:height_end + 1,
                      width_start:width_end + 1] for volume in volumes]
    if non_list:
        return volumes[0]
    return volumes