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

# Various customize image augmentation implementations specialised in 4 dimensional tensors.
# (Channel, Depth, Height, Width).
# The expected range should be between 0 and 1.


def sim_low_res(tensor, scale=2):
    """
    Simulate low resolution by down-sampling using nearest-neighbor interpolation and then up-sampling using cubic 
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
    tensor = (tensor ** gamma) * gain
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
    mean = torch.mean(tensor)
    tensor = (contrast * tensor + (1.0 - contrast) * mean).clamp(0, 1)
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
    tensor = (brightness * tensor).clamp(0, 1)
    return tensor

#    The sigma is automatically calculated in the way same as torchvision.transforms.functional.gaussian_blur(),
#    which is sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8\n

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


def random_rotation_3d(tensors, angle_range, plane='xy', interpolations=('bilinear', 'nearest'), expand_output=False, fill_values=(0, 0)):
    """
    Randomly rotates 3D PyTorch tensors with the same rotation.

    Args:
        tensors (list): Variable number of input tensors of shape (C, Depth, Height, Width).
        angle_range (tuple): Range of angles for rotation in degrees (min_angle, max_angle).
        plane (str): The plane which the random rotation will take place: 'xy'|'xz'|'yz'. Default: 'xy'.
        interpolations (tuple): Interpolation methods for each tensor ('nearest' or 'bilinear', default: ('bilinear', 'nearest')).
        expand_output (bool): If True, expand the output to fit the entire rotated image; otherwise, crop it (default: False).
        fill_values (tuple): Pixel fill values for areas outside the rotated image for each input tensor (default: (0, 0)).

    Returns:
        List of rotated tensors with the same rotation.
    """
    # Check if the number of tensors, interpolations, and fill_values are the same
    num_tensors = len(tensors)
    if len(interpolations) != num_tensors or len(fill_values) != num_tensors:
        raise ValueError("Number of tensors, interpolations, and fill_values should be the same.")

    # Generate random angles within the specified range
    angle = random.randrange(angle_range[0], angle_range[1], 30)

    rotated_tensors = []
    for i in range(num_tensors):
        tensor = tensors[i]

        if plane == "xz":
            tensor = tensor.transpose(1, 3)  # Transpose to (C, Width, Height, Depth)
        elif plane == "yz":
            tensor = tensor.transpose(2, 3)  # Transpose to (C, Depth, Width, Height)

        rotated_tensor = T_F.rotate(tensor, angle,
                                    interpolation=transforms.InterpolationMode.BILINEAR if interpolations[i] == 'bilinear' else transforms.InterpolationMode.NEAREST,
                                    expand=expand_output, fill=[fill_values[i]])

        if plane == "xz":
            rotated_tensor = rotated_tensor.transpose(1, 3)  # Transpose back to (C, Depth, Height, Width)
        elif plane == "yz":
            rotated_tensor = rotated_tensor.transpose(2, 3)  # Transpose back to (C, Depth, Height, Width)

        rotated_tensors.append(rotated_tensor)

    return rotated_tensors


def custom_rand_crop(tensors, depth, height, width, pad_if_needed=True, max_attempts=99, minimal_foreground=0.01):
    """
    Randomly crop a list of 3D PyTorch tensors given the desired depth, height, and width, ensuring at least 1% foreground object.\n
    Whether it contains foreground object is determined by the second tensor in the list,
    pixels with values larger or equal to 1 are considered foreground.

    Args:
        tensors (list of torch.Tensor): List of input tensors of shape (Channel, Depth, Height, Width).
        depth (int): Desired depth of the cropped tensors.
        height (int): Desired height of the cropped tensors.
        width (int): Desired width of the cropped tensors.
        pad_if_needed (bool): If True, pad the input tensors when they're smaller than any desired dimension (default: True).
        max_attempts (int): Maximum number of attempts to find a crop with at least 1% foreground object (default: 99).
        minimal_foreground (float): Proportion of desired minimal foreground pixels (default: 0.01).

    Returns:
        List of cropped tensors.
    """

    def contains_foreground(tensor):
        total_pixels = tensor.numel()
        pixels_greater_than_zero = (tensor > 0).sum().detach()

        return pixels_greater_than_zero >= (total_pixels * minimal_foreground)

    attempts = 0
    while attempts < max_attempts:
        # Reset attempts counter
        attempts += 1

        # Suppose all the tensors in the list are the exact same shape.
        c, d, h, w = tensors[0].shape

        if d < depth or h < height or w < width:
            if not pad_if_needed:
                raise ValueError("Input tensor dimensions are smaller than the crop size.")
            else:
                # Calculate the amount of padding needed for each dimension
                d_pad = max(0, depth - d)
                h_pad = max(0, height - h)
                w_pad = max(0, width - w)
                padded_list = []
                for tensor in tensors:
                    # Pad the tensor if needed
                    tensor = F.pad(tensor, (0, w_pad, 0, h_pad, 0, d_pad))
                    padded_list.append(tensor)
        else:
            padded_list = tensors

        # Randomly select a depth slice within the valid range
        d_offset = random.randint(0, padded_list[0].shape[1] - depth)
        h_offset = random.randint(0, padded_list[0].shape[2] - height)
        w_offset = random.randint(0, padded_list[0].shape[3] - width)

        cropped_tensors = []
        for tensor in padded_list:
            cropped_tensor = tensor[:, d_offset:d_offset + depth,
                                    h_offset:h_offset + height,
                                    w_offset:w_offset + width]
            cropped_tensors.append(cropped_tensor)

        # Check if the label tensor (2nd tensor) contains a foreground object
        if contains_foreground(cropped_tensors[1]):
            return cropped_tensors

    # If no suitable crop is found after max_attempts, raise an exception
    raise ValueError("Random clop failed: No suitable crop with foreground object found after {} attempts.".format(max_attempts))


def random_gradient(tensor, range=(0.5, 1.5), gamma=True):
    """
    Apply random gamma or contrast adjustment to a PyTorch tensor, with a gradient pattern.

    Args:
        tensor (torch.Tensor): Input image tensor of shape [channel, depth, height, width].
        range (tuple): Range for random contrast adjustment.
        gamma (bool): If true, will adjust image gamma, else contrast.

    Returns:
        torch.Tensor: Adjusted image tensor.
    """

    # Check if the input tensor has the correct shape
    if len(tensor.shape) != 4:
        raise ValueError("Input tensor must have shape [channel, depth, height, width]")

    # Generate a random side (left, right, top, bottom, front, back) for the gradient effect
    gradient_side = torch.randint(0, 6, size=(1,))

    # Generate a gradient tensor for gamma adjustment based on the selected side
    if gradient_side == 0:  # Left side
        gradient = torch.linspace(range[0], range[1], tensor.shape[3])
        gradient = gradient.view(1, 1, 1, -1)
    elif gradient_side == 1:  # Right side
        gradient = torch.linspace(range[0], range[1], tensor.shape[3])
        gradient = gradient.view(1, 1, 1, -1)
        gradient = gradient.flip(dims=(3,))
    elif gradient_side == 2:  # Top side
        gradient = torch.linspace(range[0], range[1], tensor.shape[2])
        gradient = gradient.view(1, 1, -1, 1)
    elif gradient_side == 3:  # Bottom side
        gradient = torch.linspace(range[0], range[1], tensor.shape[2])
        gradient = gradient.view(1, 1, -1, 1)
        gradient = gradient.flip(dims=(2,))
    elif gradient_side == 4:  # Front side
        gradient = torch.linspace(range[0], range[1], tensor.shape[1])
        gradient = gradient.view(1, -1, 1, 1)
    else:  # Back side
        gradient = torch.linspace(range[0], range[1], tensor.shape[1])
        gradient = gradient.view(1, -1, 1, 1)
        gradient = gradient.flip(dims=(1,))

    if gamma:
        adjusted_tensor = tensor ** gradient
    else:
        mean = torch.mean(tensor)
        adjusted_tensor = (gradient * tensor + (1.0 - gradient) * mean).clamp(0, 1)
        # adjusted_tensor = image_tensor * gradient

    return adjusted_tensor


def salt_and_pepper_noise(tensor, prob=0.01):
    """
    Add salt and pepper noise to a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor to which noise will be added.
        prob (float): Probability of adding 'salt' (maximum) and 'pepper' (minimum) noise to each element.

    Returns:
        torch.Tensor: Tensor with salt and pepper noise added.
    """
    noisy_tensor = tensor.clone()

    # Add salt noise
    salt_mask = torch.rand_like(tensor) < prob
    noisy_tensor[salt_mask] = 1.0

    # Add pepper noise
    pepper_mask = torch.rand_like(tensor) < prob
    noisy_tensor[pepper_mask] = 0.0

    return noisy_tensor


def exclude_border_labels(tensor, inward, outward):
    """
    Exclude the borders of the objects in the label tensor. Also transform it to sparsely annotated form.

    Args:
        tensor (torch.Tensor): Input tensor of shape (D,H,W). 0 for background, 1 for foreground
        inward (int): Size of morphological erosion.
        outward (int):  Size of morphological dilation.

    Returns:
        torch.Tensor: transformed Tensor. 0 for unlabeled (excluded label), 1 for foreground, 2 for background
    """
    array = tensor.numpy()

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

    transformed_tensor = torch.full_like(tensor, 2)
    transformed_tensor[tensor == 1] = 1  # Set foreground to 1
    transformed_tensor[excluded_regions] = 0  # Set excluded regions to 0
    return transformed_tensor


def binarisation(tensor):
    """
    A quick and dirty way to convert an instance labelled tensor to a semantic labelled tensor.
    """
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def binary_dilation_torch(input, structure=None, iterations=1, mask=None,
                          output=None):
    """
    Multidimensional binary dilation with the given structuring element.
    Similar to scipy.ndimage.binary_erosion but is for Pytorch Tensors.

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


def instance_contour_transform(input_tensor, contour_inward=0, contour_outward=1):
    """
    Transform an instance segmented map into a contour map.\n
    The contour is generated using morphological erosion and dilation.

    Args:
        input_tensor (torch.Tensor): The input instance segmented map with a shape of (D, H, W).
                                     0 is background and every other value is a distinct object.
        contour_inward (int): Size of morphological erosion. No longer used.
        contour_outward (int): Size of morphological dilation.

    Returns:
        torch.Tensor: Contour map where 0 are background or inside of objects while 1 are the boundaries.
    """
    input_tensor = input_tensor
    unique_values = torch.unique(input_tensor)
    transformed_tensor = torch.zeros_like(input_tensor, dtype=torch.bool)
    def process_object(value):
        if value == 0:
            return  # Skip background

        # Create a binary mask for the current object
        object_mask = (input_tensor == value)

        # Calculate bounding box for the object
        non_zero_indices = torch.nonzero(object_mask, as_tuple=True)
        min_indices = torch.min(non_zero_indices[0]), torch.min(non_zero_indices[1]), torch.min(non_zero_indices[2])
        max_indices = torch.max(non_zero_indices[0]), torch.max(non_zero_indices[1]), torch.max(non_zero_indices[2])

        # Add padding for dilation
        pad = contour_outward
        min_indices = torch.clamp(torch.tensor([min_indices[0] - pad, min_indices[1] - pad, min_indices[2] - pad]), 0)
        max_indices = torch.clamp(torch.tensor([max_indices[0] + pad+1, max_indices[1] + pad+1, max_indices[2] + pad+1]),
                                  None,
                                  torch.tensor([input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]]))

        # Crop the map according to the bounding box
        cropped_object = object_mask[min_indices[0]:max_indices[0],
                                     min_indices[1]:max_indices[1],
                                     min_indices[2]:max_indices[2]]

        # Perform dilation on the cropped object
        dilated_object = binary_dilation_torch(cropped_object, iterations=contour_outward)

        # Mark boundary area as one
        excluded_regions = (cropped_object == 0) & (dilated_object == 1)

        transformed_tensor[min_indices[0]:max_indices[0],
                           min_indices[1]:max_indices[1],
                           min_indices[2]:max_indices[2]] += excluded_regions

    # Use joblib to parallelize the loop
    Parallel(backend='threading', n_jobs=-1)(delayed(process_object)(value) for value in unique_values)
    transformed_tensor = transformed_tensor.to(torch.uint8)
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
