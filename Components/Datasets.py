import bisect
import math
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from Components.Augmentations import apply_aug, apply_aug_unsupervised, binarisation
from Components.Utils import make_label_pair_tv, path_to_array, make_path_list_predict, calculate_val_start_end, \
    calculate_predict_start_end, get_contour_maps, compute_eligibility_masks, validate_for_batch_size


class TrainDataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the training dataset.

    Args:
        images_dir (str): Path to the directory where images are stored.
        augmentation_csv (str): Path to the .csv file that contains the image augmentations parameters.
        hw_size (int): The height and width of each generated patch. Default: 64
        d_size (int): The depth of each generated patch. Default: 64
        instance_mode (bool): If true, will prepare contour tensors for instance segmentation. Default: False.
        contour_map_width (int): Width of the contour map. Default: 1.
        hdf5_key (str): If trying to load from a hdf5 file, will load the File object with this name.
    """

    def __init__(self, images_dir, augmentation_csv, hw_size=64, d_size=64,
                 instance_mode=False, contour_map_width=1, hdf5_key='Default'):
        super().__init__()
        # Get a list of file paths for images and labels
        self.file_list = np.array(make_label_pair_tv(images_dir))
        self.num_files = len(self.file_list)
        self.instance_mode = instance_mode
        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.hw_size = hw_size
        self.d_size = d_size
        if instance_mode:
            self.contour_tensors = [torch.from_numpy(get_contour_maps(item[1], 'generated_contour_maps', contour_map_width)) for item in self.file_list]
        self.lab_tensors = [torch.from_numpy(binarisation(path_to_array(item[1], key=hdf5_key, label=True))) for item in self.file_list]
        self.img_tensors = [torch.from_numpy(path_to_array(item[0], key=hdf5_key, label=False)) for item in self.file_list]
        if instance_mode:
            fg_proportions = [i.sum().item()/i.numel() for i in self.contour_tensors]
        else:
            fg_proportions = [i.sum().item()/i.numel() for i in self.lab_tensors]
        log_size = [math.log(i.numel()) for i in self.img_tensors] # Relative size of each image file in log
        masks = compute_eligibility_masks(fg_proportions, log_size)
        self.fg_proportions = masks["fg_proportions"]
        self.eligible_positive = masks["eligible_positive"]
        self.positive_log_size = masks["positive_log_size"]
        self.eligible_negative = masks["eligible_negative"]
        self.negative_log_size = masks["negative_log_size"]
        self.eligible_mixed = masks["eligible_mixed"]
        self.mixed_log_size = masks["mixed_log_size"]


    def get_statistics(self):
        return {
            "num_files": int(self.num_files),
            "fg_proportions": list(self.fg_proportions),
            "num_eligible_positive": int(len(self.eligible_positive)),
            "num_eligible_negative": int(len(self.eligible_negative)),
            "num_eligible_mixed": int(len(self.eligible_mixed)),
        }

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        """
        negative_control (str or None): If 'negative', allow the returned lab_tensor to be fully negative (no foreground object).
                                        If 'positive', allow the returned lab_tensor to be fully positive.
                                        If None, will try to generate lab_tensor that contain both positive and negative pixels. Default: None
        """
        file_idx, negative_control = idx
        background_threshold = 0.01
        foreground_threshold = 0.01
        if negative_control == 'positive':
            foreground_threshold = 0.01
            background_threshold = 0
        elif negative_control == 'negative':
            background_threshold = 0.01
            foreground_threshold = 0
        img_tensor, lab_tensor = self.img_tensors[file_idx][None, :], self.lab_tensors[file_idx][None, :]
        if self.instance_mode:
            contour_tensor = self.contour_tensors[file_idx][None, :]
            img_tensor, lab_tensor, contour_tensor = apply_aug(img_tensor, lab_tensor, contour_tensor,
                                                               self.augmentation_params, self.hw_size, self.d_size,
                                                               foreground_threshold, background_threshold)
            return img_tensor, lab_tensor, contour_tensor
        else:
            img_tensor, lab_tensor = apply_aug(img_tensor, lab_tensor, None,
                                               self.augmentation_params, self.hw_size, self.d_size,
                                               foreground_threshold, background_threshold)
            return img_tensor, lab_tensor



class UnsupervisedDataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the unsupervised training dataset.

    Args:
        images_dir (str): Path to the directory where images are stored.
        augmentation_csv (str): Path to the .csv file that contains the image augmentations parameters.
        hw_size (int): The height and width of each generated patch. Default: 64
        d_size (int): The depth of each generated patch. Default: 64
        hdf5_key (str): If trying to load from a hdf5 file, will load the File object with this name.
    """
    def __init__(self, images_dir, augmentation_csv, hw_size=64, d_size=64, hdf5_key='Default'):
        super().__init__()
        self.file_list = np.array(make_path_list_predict(images_dir))
        self.num_files = len(self.file_list)
        self.img_tensors = [torch.from_numpy(path_to_array(item, key=hdf5_key, label=False)) for item in self.file_list]
        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.hw_size = hw_size
        self.d_size = d_size
        self.log_size = [math.log(i.numel()) for i in self.img_tensors]

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        img_tensor = self.img_tensors[idx][None, :]
        img_tensor = apply_aug_unsupervised(img_tensor, self.augmentation_params, self.hw_size, self.d_size)
        return (img_tensor,)


class ValDataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the validation dataset.
    Note if the image is larger than the size specified in the augmentation_csv, it will get cropped into several
    (potentially) overlapping smaller images. The validation result will be the average of these smaller images.

    Args:
        images_dir (str): Path to the directory where images are stored.
        hw_size (int): The height and width of each generated patch.
        d_size (int): The depth of each generated patch.
        instance_mode (bool): If true, will prepare contour tensors for instance segmentation.
        contour_map_width (int): Width of the contour map. Default: 1.
        hdf5_key (str): If trying to load from a hdf5 file, will load the File object with this name.
    """

    def __init__(self, images_dir, hw_size, d_size, instance_mode, contour_map_width=1, hdf5_key='Default'):
        # Get a list of file paths for images and labels
        file_list = np.array(make_label_pair_tv(images_dir))
        self.num_files = len(file_list)
        self.instance_mode = instance_mode
        if instance_mode:
            tensor_pairs = [((path_to_array(str(item[0]), key=hdf5_key, label=False)),
                             binarisation(path_to_array(str(item[1]), label=True)),
                             get_contour_maps(item[1], 'generated_contour_maps', contour_map_width)) for item in file_list]
        else:
            tensor_pairs = [(path_to_array(str(item[0]), key=hdf5_key, label=False),
                            binarisation(path_to_array(str(item[1]), label=True))) for item in file_list]
        self.chopped_array_pairs = []
        self.total_patches = 0
        # Crop the tensors, so they are the standard shape specified in the augmentation csv.
        for pairs in tensor_pairs:
            depth, height, width = pairs[0].shape
            depth_multiplier = math.ceil(depth / d_size)
            height_multiplier = math.ceil(height / hw_size)
            width_multiplier = math.ceil(width / hw_size)
            total_multiplier = depth_multiplier * height_multiplier * width_multiplier
            self.total_patches += total_multiplier
            self.leave_out_list = []
            for depth_idx, height_idx, width_idx in product(range(depth_multiplier),
                                                            range(height_multiplier),
                                                            range(width_multiplier)):
                depth_start, depth_end = calculate_val_start_end(depth_multiplier, d_size, depth, depth_idx)
                height_start, height_end = calculate_val_start_end(height_multiplier, hw_size, height, height_idx)
                width_start, width_end = calculate_val_start_end(width_multiplier, hw_size, width, width_idx)

                cropped_img = torch.from_numpy(pairs[0][depth_start:depth_end,
                                               height_start:height_end,
                                               width_start:width_end])
                cropped_lab = torch.from_numpy(pairs[1][depth_start:depth_end,
                                               height_start:height_end,
                                               width_start:width_end])

                # pad chunks that are smaller than required
                pad_d = max(0, d_size - cropped_img.shape[0])
                pad_h = max(0, hw_size - cropped_img.shape[1])
                pad_w = max(0, hw_size - cropped_img.shape[2])
                if pad_d or pad_h or pad_w:
                    pad_tuple = (0, pad_w, 0, pad_h, 0, pad_d)
                    cropped_img = F.pad(cropped_img, pad_tuple, value=0)
                    cropped_lab = F.pad(cropped_lab, pad_tuple, value=0)

                if instance_mode:
                    cropped_contour = torch.from_numpy(pairs[2][depth_start:depth_end,
                                                       height_start:height_end,
                                                       width_start:width_end])
                    if pad_d or pad_h or pad_w:
                        cropped_contour = F.pad(cropped_contour, pad_tuple, value=0)
                    self.chopped_array_pairs.append((cropped_img, cropped_lab, cropped_contour))
                else:
                    self.chopped_array_pairs.append((cropped_img, cropped_lab))
        super().__init__()

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        img_tensor, lab_tensor = self.chopped_array_pairs[idx][0][None, :], self.chopped_array_pairs[idx][1][None, :]
        if self.instance_mode:
            contour_tensor = self.chopped_array_pairs[idx][2][None, :]
            return img_tensor.to(torch.float32), lab_tensor.to(torch.float32), contour_tensor.to(torch.float32)
        else:
            return img_tensor.to(torch.float32), lab_tensor.to(torch.float32)


class PredictDataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the predict dataset.
    Creates patches on the fly to avoid storing all patches in memory.
    Each image is loaded once and kept as a padded array.

    Args:
        images_dir (str): Path to the directory where images are stored.
        hw_size (int): Height and width of the central patch (without overlap).
        depth_size (int): Depth of the central patch (without overlap).
        hw_overlap (int): Overlap in height and width (added to each side).
        depth_overlap (int): Overlap in depth (added to each side).
        hdf5_key (str): Key used when loading from HDF5 files.
    """
    def __init__(self, images_dir, hw_size=128, depth_size=128,
                 hw_overlap=16, depth_overlap=16, hdf5_key='Default'):
        self.file_list = make_path_list_predict(images_dir)
        self.hw_size = hw_size
        self.depth_size = depth_size
        self.hw_overlap = hw_overlap
        self.depth_overlap = depth_overlap
        self.hdf5_key = hdf5_key

        # Store per‑file metadata and padded images
        self.padded_images = []      # list of padded arrays (1, D_pad, H_pad, W_pad)
        self.meta_list = []          # list of (file_name, original_shape) for stitching
        self.patch_infos = []        # list of (depth_mult, height_mult, width_mult,
                                     #           orig_depth, orig_height, orig_width)
        self.cum_patch_counts = []   # cumulative number of patches per file

        total_patches = 0
        for file_path in self.file_list:
            # Load image (3D) and add channel dimension
            image = path_to_array(str(file_path), key=hdf5_key, label=False)  # (D, H, W)
            image = image[None, :]  # (1, D, H, W)
            c, depth, height, width = image.shape
            file_name = file_path.name

            # Calculate number of patches needed
            depth_mult = math.ceil(depth / depth_size)
            height_mult = math.ceil(height / hw_size)
            width_mult = math.ceil(width / hw_size)

            # Symmetric padding
            paddings = ((0, 0),
                        (depth_overlap, depth_overlap),
                        (hw_overlap, hw_overlap),
                        (hw_overlap, hw_overlap))
            padded_image = torch.from_numpy(np.pad(image, paddings, mode="symmetric"))

            # Store everything
            self.padded_images.append(padded_image)
            self.meta_list.append((file_name, image.shape))
            self.patch_infos.append((depth_mult, height_mult, width_mult,
                                     depth, height, width))
            total_patches += depth_mult * height_mult * width_mult
            self.cum_patch_counts.append(total_patches)

        self.total_patches = total_patches

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cum_patch_counts, idx)
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cum_patch_counts[file_idx - 1]

        # Retrieve metadata for that file
        depth_mult, height_mult, width_mult, depth, height, width = self.patch_infos[file_idx]

        # Reconstruct the three patch indices (order: depth, height, width)
        width_idx = local_idx % width_mult
        height_idx = (local_idx // width_mult) % height_mult
        depth_idx = local_idx // (width_mult * height_mult)

        # Compute start/end coordinates using the same function as before
        depth_start, depth_end = calculate_predict_start_end(
            depth_mult, self.depth_size, depth, depth_idx, self.depth_overlap)
        height_start, height_end = calculate_predict_start_end(
            height_mult, self.hw_size, height, height_idx, self.hw_overlap)
        width_start, width_end = calculate_predict_start_end(
            width_mult, self.hw_size, width, width_idx, self.hw_overlap)

        # Extract the patch from the pre‑padded image
        padded_img = self.padded_images[file_idx]  # shape (1, D_pad, H_pad, W_pad)
        patch = padded_img[:,
                           depth_start:depth_end,
                           height_start:height_end,
                           width_start:width_end].to(torch.float32)

        expected_d = self.depth_size + 2 * self.depth_overlap
        expected_h = self.hw_size + 2 * self.hw_overlap
        expected_w = self.hw_size + 2 * self.hw_overlap
        pad_d = max(0, expected_d - patch.shape[1])
        pad_h = max(0, expected_h - patch.shape[2])
        pad_w = max(0, expected_w - patch.shape[3])
        if pad_d or pad_h or pad_w:
            patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d), value=0)

        patch = patch.to(torch.float32)

        return patch, self.hw_overlap, self.depth_overlap

    def __getmetainfo__(self):
        """Return metadata (file name, original shape) for each original image."""
        return self.meta_list

