import imageio
import torch.nn.functional as F
import torchvision.transforms.functional as T_F
import torchvision.transforms as transforms
import random
import torch


def sim_low_res_3D(input, scale):
    """
    Simulate low resolution by downsampling using nearest-neighbor interpolation and then upsampling using cubic interpolation.

    Args:
        input (torch.Tensor): Input tensor of shape (D, H, W) or (H, W).
        scale (float): Scale factor for downsampling and upsampling.

    Returns:
        torch.Tensor: Simulated low-resolution tensor with the same shape as the input.
    """
    shape = input.shape
    input = input.unsqueeze(0).unsqueeze(0)
    input = F.interpolate(input, scale_factor=scale, mode='nearest')
    if len(shape) == 3:
        input = F.interpolate(input, size=[shape[0], shape[1], shape[2]], mode='trilinear')
    if len(shape) == 2:
        input = F.interpolate(input, size=[shape[0], shape[1]], mode='bicubic')
    return input.squeeze(0).squeeze(0)


def adj_gamma_3D(input, gamma, gain=1):
    """
    Adjust gamma correction for a 3D tensor.

    Args:
        input (torch.Tensor): Input 3D tensor.
        gamma (float): Non-negative gamma correction factor.
        gain (float, optional): Multiplicative gain. Default is 1.

    Returns:
        torch.Tensor: Gamma-adjusted 3D tensor with the same shape as the input.
    """
    img = [T_F.adjust_gamma(i, gamma, gain) for i in torch.split(input, dim=0, split_size_or_sections=1)]
    img = torch.cat(img, dim=0)
    return img


def adj_contrast_3d(input, contrast):
    """
    Adjust contrast for a 3D tensor.

    Args:
        input (torch.Tensor): Input 3D tensor.
        contrast (float): Contrast adjustment factor.

    Returns:
        torch.Tensor: Contrast-adjusted 3D tensor with the same shape as the input.
    """
    three_d_stacks = [i for i in torch.split(input, dim=0, split_size_or_sections=1)]
    adjusted_stacks = []
    for stack in three_d_stacks:
        adjusted_two_d_slices = T_F.adjust_contrast(stack, contrast)
        adjusted_stacks.append(adjusted_two_d_slices)
    output = torch.cat(adjusted_stacks, dim=0)
    return output


def gaussian_blur_3d(input, kernel):
    """
    Apply Gaussian blur to a 3D tensor.

    Args:
        input (torch.Tensor): Input 3D tensor.
        kernel (List[List[float]]): 2D Gaussian kernel for the blur.

    Returns:
        torch.Tensor: 3D tensor after applying Gaussian blur with the same shape as the input.
    """
    input = [T_F.gaussian_blur(i, kernel) for i in torch.split(input, dim=0, split_size_or_sections=1)]
    input = torch.cat(input, dim=0)
    return input


def random_rotation_3d(tensor1, tensor2, angle_range, plane='xy', interpolation1='bilinear',
                       interpolation2='nearest', expand_output=False, fill_value_1=0, fill_value_2=0):
    """
    Randomly rotates two 3D PyTorch tensors with the same rotation.

    Args:
        tensor1 (torch.Tensor): First input tensor of shape (Depth, Height, Width).
        tensor2 (torch.Tensor): Second input tensor of shape (Depth, Height, Width).
        angle_range (tuple): Range of angles for rotation in degrees (min_angle, max_angle).
        plane (str): The plane which the random rotation will take place (default: 'xy').
        interpolation1 (str): Interpolation method for tensor1 ('nearest' or 'bilinear', default: 'bilinear').
        interpolation2 (str): Interpolation method for tensor2 ('nearest' or 'bilinear', default: 'nearest').
        expand_output (bool): If True, expand the output to fit the entire rotated image; otherwise, crop it (default: False).
        fill_value_1 (float): Pixel fill value for areas outside the rotated image for the first input tensor (default: 0).
        fill_value_2 (float): Pixel fill value for areas outside the rotated image for the second input tensor (default: 0).

    Returns:
        Tuple of two rotated tensors with the same rotation.
    """
    # Generate random angles within the specified range
    angle = random.uniform(angle_range[0], angle_range[1])

    if plane == "xz":
        tensor1, tensor2 = tensor1.transpose(0, 2), tensor2.transpose(0, 2)  # Transpose to (Width, Height, Depth)
    elif plane == "yz":
        tensor1, tensor2 = tensor1.transpose(1, 2), tensor2.transpose(1, 2)  # Transpose to (Depth, Width, Height)

    tensor1 = T_F.rotate(tensor1, angle,
                         interpolation=transforms.InterpolationMode.BILINEAR if interpolation1 == 'bilinear' else transforms.InterpolationMode.NEAREST,
                         expand=expand_output, fill=fill_value_1)
    tensor2 = T_F.rotate(tensor2, angle,
                         interpolation=transforms.InterpolationMode.BILINEAR if interpolation2 == 'bilinear' else transforms.InterpolationMode.NEAREST,
                         expand=expand_output, fill=fill_value_2)

    if plane == "xz":
        tensor1, tensor2 = tensor1.transpose(0, 2), tensor2.transpose(0, 2)  # Transpose back to (Depth, Height, Width)
    elif plane == "yz":
        tensor1, tensor2 = tensor1.transpose(1, 2), tensor2.transpose(1, 2)  # Transpose back to (Depth, Height, Width)

    return tensor1, tensor2


def custom_rand_crop(tensor_list, depth, height, width, pad_if_needed=True, max_attempts=99):
    """
    Randomly crop a list of 3D PyTorch tensors given the desired depth, height, and width, ensuring at least 2% foreground object.

    Args:
        tensor_list (list of torch.Tensor): List of input tensors of shape (Channel, Depth, Height, Width).
        depth (int): Desired depth of the cropped tensors.
        height (int): Desired height of the cropped tensors.
        width (int): Desired width of the cropped tensors.
        pad_if_needed (bool): If True, pad the input tensors when they're smaller than any desired dimension (default: True).
        max_attempts (int): Maximum number of attempts to find a crop with at least one foreground object (default: 99).

    Returns:
        List of cropped tensors.
    """

    def contains_foreground(tensor):
        # Check if the tensor contains at least 2% pixels higher than 0.
        total_pixels = tensor.numel()
        pixels_greater_than_zero = (tensor > 0).sum().item()

        return pixels_greater_than_zero >= (total_pixels * 0.02)

    attempts = 0
    while attempts < max_attempts:
        # Reset attempts counter
        attempts += 1

        # Suppose all the tensors in the list are the exact same shape.
        c, d, h, w = tensor_list[0].shape

        if d < depth or h < height or w < width:
            if not pad_if_needed:
                raise ValueError("Input tensor dimensions are smaller than the crop size.")
            else:
                # Calculate the amount of padding needed for each dimension
                d_pad = max(0, depth - d)
                h_pad = max(0, height - h)
                w_pad = max(0, width - w)
                padded_list = []
                for tensor in tensor_list:
                    # Pad the tensor if needed
                    tensor = F.pad(tensor, (0, w_pad, 0, h_pad, 0, d_pad))
                    padded_list.append(tensor)
        else:
            padded_list = tensor_list

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
    raise ValueError("No suitable crop with foreground object found after {} attempts.".format(max_attempts))


if __name__ == "__main__":
    pass