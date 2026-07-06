import random
import torch
import math
import scipy

import numpy as np
import torch.nn.functional as F

from joblib import Parallel, delayed
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as T_F

device = "cuda" if torch.cuda.is_available() else "cpu"
# Various customize image augmentation implementations specialised in 4 dimensional tensors.
# (Channel, Depth, Height, Width).


def sim_low_res(tensor, scale=2):
    """
    Simulate low resolution by down-sampling using nearest-neighbor interpolation and then up-sampling using cubic/linear
    interpolation.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W).
        scale (float): Scale factor for down-sampling and up-sampling. Default is 2.

    Returns:
        torch.Tensor: Simulated low-resolution tensor with the same shape as the input.
    """
    shape = tensor.shape
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, scale_factor=scale, mode='nearest-exact')
    tensor = F.interpolate(tensor, size=[shape[1], shape[2], shape[3]], mode='trilinear')
    return tensor.squeeze(0)


def adj_gamma(tensor, gamma, gain=1):
    """
    Adjust gamma correction.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W).
        gamma (float): Non-negative gamma correction factor.
        gain (float): Multiplicative gain. Default is 1.

    Returns:
        torch.Tensor: Gamma-adjusted 3D tensor.
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
    Adjust contrast for a tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W).
        contrast (float): Contrast adjustment factor.

    Returns:
        torch.Tensor: Contrast-adjusted 3D tensor.
    """
    tensor *= contrast
    return tensor


def adj_brightness(tensor, brightness):
    """
    Adjust brightness for a tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W).
        brightness (float): Brightness adjustment factor.

    Returns:
        torch.Tensor: Brightness-adjusted 3D tensor.
    """
    return torch.clamp(brightness + tensor)


def gaussian_blur_3d(tensor, kernel_size=3, sigma=0.8):
    """
    Apply 3D Gaussian blur to a tensor.
    High kernel_size could slow down augmentation significantly.

    Args:
        tensor (torch.Tensor): Input tensor of shape {C, D, H, W}.
        kernel_size (int): Size of the Gaussian kernel. Default: 3.
        sigma (float): Sigma of the Gaussian blur. Default: 0.8.

    Returns:
        torch.Tensor
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
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(input_channels, input_channels, 1, 1, 1).to(torch.float32)

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
    grid = F.affine_grid(affine_4x4[:3, :].unsqueeze(0), tensor.unsqueeze(0).shape, align_corners=False).to(torch.float32)

    # Apply the grid sample using the specified interpolation
    rotated = F.grid_sample(tensor.to(torch.float32).unsqueeze(0), grid, mode=interpolation, padding_mode='reflection', align_corners=False)

    return rotated.squeeze(0)


def custom_rand_crop_rotate(tensors, depth, height, width,
                            angle_range=((0, 360), (0, 360), (0, 360)),
                            planes=['xy'], interpolations=('bilinear', 'nearest'),  # fill_values=(0, 0),
                            ensure_bothground=True, max_attempts=50, minimal_foreground=0.01, minimal_background=0.02,
                            zarr=False, img_mean=0, img_std=1):
    """
    Randomly crop then rotate a list of 3D PyTorch tensors given the desired depth, height, and width,
    preferably within the required foreground or background proportion.\n
    The foreground object is determined by the second tensor in the list,
    pixels with values larger or equal to 1 are considered foreground.\n
    If no foreground object is found after max_attempts attempts, it will output a warning message and crop a random volume.

    Args:
        tensors (list of torch.Tensor): List of input tensors of shape (Channel, Depth, Height, Width).
        depth (int): Desired depth of the cropped tensors.
        height (int): Desired height of the cropped tensors.
        width (int): Desired width of the cropped tensors.
        angle_range (list of tuple): Range of rotation angles (min_angle, max_angle) for each plane in degrees.
        planes (list or None): The plane(s) on which the random rotation will be applied: ['xy', 'xz', 'yz']. Default: ['xy']
        interpolations (tuple): Interpolation methods for each tensor ('nearest' or 'bilinear').
        ensure_bothground (bool): If True, will try to ensure that the output contains both foreground and background objects according to the settings below. (default: True)
        max_attempts (int): Maximum number of attempts to find a crop which satisfy the required foreground/background condition (default: 50).
        minimal_foreground (float): Proportion of desired minimal foreground pixels (default: 0.01).
        minimal_background (float): Proportion of desired minimal background pixels (default: 0.02).
        zarr (bool): Needs to be set to true if the input are not torch tensor but zarr arrays. (default: False)
        img_mean (float): Used if zarr is True, for normalisation of the input volume. (default: 0)
        img_std (float): Used if zarr is True, for normalisation of the input volume. (default: 1)

    Returns:
        List of cropped tensors.
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
            if zarr:
                tensor = np.pad(tensor, ((0,0), (0,d_pad), (0,h_pad), (0,w_pad)))
            else:
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


def random_gradient(tensor, range=(0.5, 1.5), mode='gamma'):
    """
    Apply random gamma or contrast or brightness adjustment to a PyTorch tensor, with a gradient pattern.

    Args:
        tensor (torch.Tensor): Input image tensor of shape [channel, depth, height, width].
        range (tuple): Range for random contrast adjustment.
        mode (str): 'gamma' or 'contrast' or 'brightness'.

    Returns:
        torch.Tensor: Adjusted image tensor.
    """

    a, b = random.uniform(range[0], range[1]), random.uniform(range[0], range[1])
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
        torch.Tensor
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
    Exclude the borders of the objects in the label array. Also transform it to sparsely annotated form.

    Args:
        array (np.Array): Input array of shape (D,H,W). 0 for background, 1 for foreground
        inward (int): Size of morphological erosion.
        outward (int):  Size of morphological dilation.

    Returns:
        torch.Tensor: transformed array. 0 for unlabeled (excluded label), 1 for foreground, 2 for background
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

    output_tensor = input_tensor.to(torch.int64)[:, :,
                                                depth_indices[:, None, None],
                                                height_indices[None, :, None],
                                                width_indices[None, None, :]].to(input_tensor.dtype)

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
        # Since we only deal with images of single channel, no need to worry the channel dim got downscaled as well.
        shape = [max(int(shape / 2**octave), 1) for shape in origin_shape]
        noise = (torch.randn(shape, dtype=torch.float32) * random.uniform(0.5, 1.5) * (0.5**octave) * strength).unsqueeze(0)
        noises += F.interpolate(noise, origin_shape[1:], mode='trilinear', align_corners=False).squeeze(0)
    input_tensor += noises
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


def apply_aug(img_tensor, lab_tensor, contour_tensor, augmentation_params,
              hw_size, d_size, foreground_threshold, background_threshold, zarr=False, img_mean=0, img_std=1):
    """
    Apply Image Augmentations to an image tensor and its label tensor using the augmentation parameters from a DataFrame.
    Can have an optional contour tensor processed as well.

    Args:
        img_tensor (np.Array): Image tensor, should be float32.
        lab_tensor (np.Array): Label tensor, should be the same shape as img_tensor. Bool.
        contour_tensor (torch.Tensor or None): Optional Contour tensor, should be the same shape as img_tensor. Bool.
        augmentation_params (DataFrame): The DataFrame which the augmentation parameters will be used from.
        hw_size (int): The height and width of each generated patch.
        d_size (int): The depth of each generated patch.
        foreground_threshold (float): Proportion of desired minimal foreground pixels in the produced label tensor.
        background_threshold (float): Proportion of desired minimal background pixels in the produced label tensor.
        zarr (bool): To indicate if the input tensor is not a numpy array but a zarr array. (default: False)
        img_mean (float): Used if zarr is True, for normalisation of the input volume. (default: 0)
        img_std (float): Used if zarr is True, for normalisation of the input volume. (default: 1)

    Returns:
        Transformed Image and Label Tensor.
    """
    rotation_methods = []
    rotation_angles = []
    for _, row in augmentation_params.iterrows():
        augmentation_method, prob = row['Augmentation'], row['Probability']
        if augmentation_method == 'Rotate xy' and random.random() < prob:
            rotation_methods.append('xy')
            rotation_angles.append((row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate xz' and random.random() < prob:
            rotation_methods.append('xz')
            rotation_angles.append((row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate yz' and random.random() < prob:
            rotation_methods.append('yz')
            rotation_angles.append((row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rescaling':
            if random.random() < prob:
                scale = random.uniform(row['Low Bound'], row['High Bound'])
                if contour_tensor is not None:
                    img_tensor, lab_tensor, contour_tensor = custom_rand_crop_rotate(
                        [img_tensor, lab_tensor, contour_tensor],
                        int(scale * d_size), int(scale * hw_size), int(scale * hw_size),
                        rotation_angles, rotation_methods,
                        ('bilinear', 'nearest', 'nearest'),
                        minimal_foreground=foreground_threshold,
                        minimal_background=background_threshold,
                        zarr=zarr, img_mean=img_mean, img_std=img_std)
                    contour_tensor = contour_tensor[None, :]
                    contour_tensor = F.interpolate(contour_tensor, size=(d_size, hw_size, hw_size),
                                                   mode="nearest-exact")
                    contour_tensor = torch.squeeze(contour_tensor, 0)
                else:
                    img_tensor, lab_tensor = custom_rand_crop_rotate([img_tensor, lab_tensor],
                                                                         int(scale * d_size),
                                                                         int(scale * hw_size),
                                                                         int(scale * hw_size),
                                                                         rotation_angles, rotation_methods,
                                                                         ('bilinear', 'nearest'),
                                                                         minimal_foreground=foreground_threshold,
                                                                         minimal_background=background_threshold,
                                                                         zarr=zarr, img_mean=img_mean, img_std=img_std)
                img_tensor = img_tensor[None, :]
                img_tensor = F.interpolate(img_tensor, size=(d_size, hw_size, hw_size), mode="trilinear",
                                           align_corners=True)
                lab_tensor = lab_tensor[None, :]
                if lab_tensor.dtype != torch.uint8:
                    lab_tensor = nearest_interpolate(lab_tensor, (d_size, hw_size, hw_size))
                else:
                    lab_tensor = F.interpolate(lab_tensor, size=(d_size, hw_size, hw_size), mode="nearest-exact")
                img_tensor = torch.squeeze(img_tensor, 0)
                lab_tensor = torch.squeeze(lab_tensor, 0)
            else:
                if contour_tensor is not None:
                    img_tensor, lab_tensor, contour_tensor = custom_rand_crop_rotate(
                        [img_tensor, lab_tensor, contour_tensor],
                        d_size, hw_size, hw_size,
                        rotation_angles, rotation_methods,
                        ('bilinear', 'nearest', 'nearest'),
                        minimal_foreground=foreground_threshold, minimal_background=background_threshold,
                        zarr=zarr, img_mean=img_mean, img_std=img_std)
                else:
                    img_tensor, lab_tensor = custom_rand_crop_rotate(
                        [img_tensor, lab_tensor],
                        d_size, hw_size, hw_size,
                        rotation_angles, rotation_methods,
                        ('bilinear', 'nearest'),
                        minimal_foreground=foreground_threshold, minimal_background=background_threshold,
                        zarr=zarr, img_mean=img_mean, img_std=img_std)
        elif augmentation_method == 'Edge Replicate Pad' and random.random() < prob:
            if contour_tensor is not None:
                img_tensor, lab_tensor, contour_tensor = edge_replicate_pad((img_tensor, lab_tensor, contour_tensor), row['Value'])
            else:
                img_tensor, lab_tensor = edge_replicate_pad((img_tensor, lab_tensor), row['Value'])
        elif augmentation_method == 'Vertical Flip' and random.random() < prob:
            img_tensor, lab_tensor = T_F.vflip(img_tensor), T_F.vflip(lab_tensor)
            if contour_tensor is not None:
                contour_tensor = T_F.vflip(contour_tensor)
        elif augmentation_method == 'Horizontal Flip' and random.random() < prob:
            img_tensor, lab_tensor = T_F.hflip(img_tensor), T_F.hflip(lab_tensor)
            if contour_tensor is not None:
                contour_tensor = T_F.hflip(contour_tensor)
        elif augmentation_method == 'Depth Flip' and random.random() < prob:
            img_tensor, lab_tensor = img_tensor.flip([1]), lab_tensor.flip([1])
            if contour_tensor is not None:
                contour_tensor = contour_tensor.flip([1])
        elif augmentation_method == 'Simulate Low Resolution' and random.random() < prob:
            img_tensor = sim_low_res(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Noise' and random.random() < prob:
            img_tensor = gaussian_noise(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Blur' and random.random() < prob:
            img_tensor = gaussian_blur_3d(img_tensor, int(row['Value']),
                                              random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gradient Gamma' and random.random() < prob:
            img_tensor = random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'gamma')
        elif augmentation_method == 'Gradient Contrast' and random.random() < prob:
            img_tensor = random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'contrast')
        elif augmentation_method == 'Gradient Brightness' and random.random() < prob:
            img_tensor = random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'brightness')
        elif augmentation_method == 'Adjust Gamma' and random.random() < prob:
            img_tensor = adj_gamma(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Contrast' and random.random() < prob:
            img_tensor = adj_contrast(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Brightness' and random.random() < prob:
            img_tensor = adj_brightness(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Salt And Pepper' and random.random() < prob:
            img_tensor = salt_and_pepper_noise(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Label Blur' and random.random() < prob:
            lab_tensor = gaussian_blur_3d(lab_tensor, int(row['Value']),
                                              random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Contour Blur' and random.random() < prob and contour_tensor is not None:
            contour_tensor = gaussian_blur_3d(contour_tensor, int(row['Value']),
                                                  random.uniform(row['Low Bound'], row['High Bound']))
    if contour_tensor is not None:
        return img_tensor, lab_tensor, contour_tensor
    else:
        return img_tensor, lab_tensor


def apply_aug_unsupervised(img_tensor, augmentation_params, hw_size, d_size, zarr=False, img_mean=0, img_std=1):
    """
        Apply Image Augmentations to an image tensor using the augmentation parameters from a DataFrame.

        Args:
            img_tensor (torch.Tensor) Image tensor, should be float32.
            augmentation_params (DataFrame): The DataFrame which the augmentation parameters will be used from.
            hw_size (int): The height and width of each generated patch.
            d_size (int): The depth of each generated patch.
            zarr (bool): To indicate if the input tensor is not a numpy array but a zarr array. (default: False)
            img_mean (float): Used if zarr is True, for normalisation of the input volume. (default: 0)
            img_std (float): Used if zarr is True, for normalisation of the input volume. (default: 1)

        Returns:
            Transformed Image Tensor.
        """
    rotation_methods = []
    rotation_angles = []
    for _, row in augmentation_params.iterrows():
        augmentation_method, prob = row['Augmentation'], row['Probability']
        if augmentation_method == 'Rotate xy' and random.random() < prob:
            rotation_methods.append('xy')
            rotation_angles.append((row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate xz' and random.random() < prob:
            rotation_methods.append('xz')
            rotation_angles.append((row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate yz' and random.random() < prob:
            rotation_methods.append('yz')
            rotation_angles.append((row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rescaling':
            if random.random() < prob:
                scale = random.uniform(row['Low Bound'], row['High Bound'])
                img_tensor = custom_rand_crop_rotate(
                    [img_tensor],
                    int(scale * d_size),
                    int(scale * hw_size),
                    int(scale * hw_size),
                    rotation_angles, rotation_methods,
                    ('bilinear',),
                    ensure_bothground=False, zarr=zarr, img_mean=img_mean, img_std=img_std)
                img_tensor = img_tensor[0]
                img_tensor = img_tensor[None, :]
                img_tensor = F.interpolate(img_tensor, size=(d_size, hw_size, hw_size), mode="trilinear",
                                           align_corners=True)
                img_tensor = torch.squeeze(img_tensor, 0)
            else:
                img_tensor = custom_rand_crop_rotate(
                    [img_tensor],
                    d_size, hw_size, hw_size,
                    rotation_angles, rotation_methods,
                    ('bilinear',),
                    ensure_bothground=False, zarr=zarr, img_mean=img_mean, img_std=img_std)
                img_tensor = img_tensor[0]
        elif augmentation_method == 'Edge Replicate Pad' and random.random() < prob:
            img_tensor = edge_replicate_pad((img_tensor,), row['Value'])
        elif augmentation_method == 'Vertical Flip' and random.random() < prob:
            img_tensor = T_F.vflip(img_tensor)
        elif augmentation_method == 'Horizontal Flip' and random.random() < prob:
            img_tensor = T_F.hflip(img_tensor)
        elif augmentation_method == 'Depth Flip' and random.random() < prob:
            img_tensor = img_tensor.flip([1])
        elif augmentation_method == 'Simulate Low Resolution' and random.random() < prob:
            img_tensor = sim_low_res(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Noise' and random.random() < prob:
            img_tensor = gaussian_noise(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Blur' and random.random() < prob:
            img_tensor = gaussian_blur_3d(img_tensor, int(row['Value']),
                                              random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gradient Gamma' and random.random() < prob:
            img_tensor = random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'gamma')
        elif augmentation_method == 'Gradient Contrast' and random.random() < prob:
            img_tensor = random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'contrast')
        elif augmentation_method == 'Gradient Brightness' and random.random() < prob:
            img_tensor = random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'brightness')
        elif augmentation_method == 'Adjust Gamma' and random.random() < prob:
            img_tensor = adj_gamma(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Contrast' and random.random() < prob:
            img_tensor = adj_contrast(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Brightness' and random.random() < prob:
            img_tensor = adj_brightness(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Salt And Pepper' and random.random() < prob:
            img_tensor = salt_and_pepper_noise(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
    return img_tensor
