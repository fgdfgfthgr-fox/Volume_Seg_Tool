import gc
import math
import os

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms.v2.functional as T_F
import numpy as np
import imageio
import random
import time
import h5py
import multiprocessing
import tracemalloc
import pandas as pd
from multiprocessing import Pool, shared_memory, cpu_count
import skimage.morphology as morph
from scipy.ndimage import label
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS
from . import Augmentations as Aug
from . import MorphologicalFunctions as Morph

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_label_fname(fname):
    return 'Labels_' + fname



def path_to_array(path, label=False, key='default', norm_strategy='std'):
    """
    Transform a path to an image file into a Pytorch tensor. Can support other image format but usually tif is the only one that can store 3d information.

    Args:
        path (str): Path to the image file.
        label (bool): If false, the output will be normalized. Default: False.
        key (str): If trying to load from a hdf5 file, will load the File object with this name.
        norm_strategy (str): The strategy used to normalize the image (when isn't a label).
        Can be 'std' to minus by mean then divide by std. Or 'n1tp1' to put it between negative and positive one. Default: 'std'.

    Returns:
        torch.Tensor: transformed Tensor.
    """
    if 'hdf5' in path:
        file_object = h5py.File(path, 'r')
        img = np.array(file_object[key])
    else:
        # ToTensor()对16位图不方便，因此才用这招
        img = imageio.v3.imread(path)
    if label:
        img_max = img.max()
        if len(np.unique(img)) <= 2:
            new_dtype = np.bool_
        elif img_max <= np.iinfo(np.uint8).max:
            new_dtype = np.uint8
        elif img_max <= np.iinfo(np.uint16).max:
            new_dtype = np.uint16
        else: # I don't think anyone will use more than the maximum value of uint32...
            new_dtype = np.uint32
        img = img.astype(new_dtype, copy=False)
    else:
        non_zero = img[img != 0]
        if norm_strategy == 'std':
            mean, std = np.mean(non_zero, dtype=np.float32), np.std(non_zero, dtype=np.float32) + 1e-8
            # Take less memory than img = (img - mean) / std
            img = img.astype(np.float32, copy=False)
            img -= mean
            np.divide(img, std, img)
        elif norm_strategy == 'n1tp1':
            img = img.astype(np.float32, copy=False)
            q1, q99 = np.percentile(non_zero, [1, 99])
            # To 0 and 1
            img -= q1
            img /= (q99-q1)
            np.clip(img, 0, 1, img)
            # To -1 and 1
            img -= 0.5
            img *= 2
    return img


def apply_aug(img_tensor, lab_tensor, contour_tensor, augmentation_params,
              hw_size, d_size, foreground_threshold, background_threshold):
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
                    img_tensor, lab_tensor, contour_tensor = Aug.custom_rand_crop_rotate(
                        [img_tensor, lab_tensor, contour_tensor],
                        int(scale * d_size), int(scale * hw_size), int(scale * hw_size),
                        rotation_angles, rotation_methods,
                        ('bilinear', 'nearest', 'nearest'),
                        minimal_foreground=foreground_threshold,
                        minimal_background=background_threshold)
                    contour_tensor = contour_tensor[None, :]
                    contour_tensor = F.interpolate(contour_tensor, size=(d_size, hw_size, hw_size),
                                                   mode="nearest-exact")
                    contour_tensor = torch.squeeze(contour_tensor, 0)
                else:
                    img_tensor, lab_tensor = Aug.custom_rand_crop_rotate([img_tensor, lab_tensor],
                                                                         int(scale * d_size),
                                                                         int(scale * hw_size),
                                                                         int(scale * hw_size),
                                                                         rotation_angles, rotation_methods,
                                                                         ('bilinear', 'nearest'),
                                                                         minimal_foreground=foreground_threshold,
                                                                         minimal_background=background_threshold)
                img_tensor = img_tensor[None, :]
                img_tensor = F.interpolate(img_tensor, size=(d_size, hw_size, hw_size), mode="trilinear",
                                           align_corners=True)
                lab_tensor = lab_tensor[None, :]
                if lab_tensor.dtype != torch.uint8:
                    lab_tensor = Aug.nearest_interpolate(lab_tensor, (d_size, hw_size, hw_size))
                else:
                    lab_tensor = F.interpolate(lab_tensor, size=(d_size, hw_size, hw_size), mode="nearest-exact")
                img_tensor = torch.squeeze(img_tensor, 0)
                lab_tensor = torch.squeeze(lab_tensor, 0)
            else:
                if contour_tensor is not None:
                    img_tensor, lab_tensor, contour_tensor = Aug.custom_rand_crop_rotate(
                        [img_tensor, lab_tensor, contour_tensor],
                        d_size, hw_size, hw_size,
                        rotation_angles, rotation_methods,
                        ('bilinear', 'nearest', 'nearest'),
                        minimal_foreground=foreground_threshold, minimal_background=background_threshold)
                else:
                    img_tensor, lab_tensor = Aug.custom_rand_crop_rotate(
                        [img_tensor, lab_tensor],
                        d_size, hw_size, hw_size,
                        rotation_angles, rotation_methods,
                        ('bilinear', 'nearest'),
                        minimal_foreground=foreground_threshold, minimal_background=background_threshold)
        elif augmentation_method == 'Edge Replicate Pad' and random.random() < prob:
            if contour_tensor is not None:
                img_tensor, lab_tensor, contour_tensor = Aug.edge_replicate_pad((img_tensor, lab_tensor, contour_tensor), row['Value'])
            else:
                img_tensor, lab_tensor = Aug.edge_replicate_pad((img_tensor, lab_tensor), row['Value'])
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
            img_tensor = Aug.sim_low_res(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Noise' and random.random() < prob:
            img_tensor = Aug.gaussian_noise(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Blur' and random.random() < prob:
            img_tensor = Aug.gaussian_blur_3d(img_tensor, int(row['Value']),
                                              random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gradient Gamma' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'gamma')
        elif augmentation_method == 'Gradient Contrast' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'contrast')
        elif augmentation_method == 'Gradient Brightness' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'brightness')
        elif augmentation_method == 'Adjust Gamma' and random.random() < prob:
            img_tensor = Aug.adj_gamma(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Contrast' and random.random() < prob:
            img_tensor = Aug.adj_contrast(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Brightness' and random.random() < prob:
            img_tensor = Aug.adj_brightness(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Salt And Pepper' and random.random() < prob:
            img_tensor = Aug.salt_and_pepper_noise(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Label Blur' and random.random() < prob:
            lab_tensor = Aug.gaussian_blur_3d(lab_tensor, int(row['Value']),
                                              random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Contour Blur' and random.random() < prob and contour_tensor is not None:
            contour_tensor = Aug.gaussian_blur_3d(contour_tensor, int(row['Value']),
                                                  random.uniform(row['Low Bound'], row['High Bound']))
    if contour_tensor is not None:
        return img_tensor, lab_tensor, contour_tensor
    else:
        return img_tensor, lab_tensor


def apply_aug_unsupervised(img_tensor, augmentation_params, hw_size, d_size):
    """
        Apply Image Augmentations to an image tensor using the augmentation parameters from a DataFrame.

        Args:
            img_tensor (torch.Tensor) Image tensor, should be float32.
            augmentation_params (DataFrame): The DataFrame which the augmentation parameters will be used from.
            hw_size (int): The height and width of each generated patch.
            d_size (int): The depth of each generated patch.

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
                img_tensor = Aug.custom_rand_crop_rotate(
                    [img_tensor],
                    int(scale * d_size),
                    int(scale * hw_size),
                    int(scale * hw_size),
                    rotation_angles, rotation_methods,
                    ('bilinear',),
                    ensure_bothground=False)
                img_tensor = img_tensor[0]
                img_tensor = img_tensor[None, :]
                img_tensor = F.interpolate(img_tensor, size=(d_size, hw_size, hw_size), mode="trilinear",
                                           align_corners=True)
                img_tensor = torch.squeeze(img_tensor, 0)
            else:
                img_tensor = Aug.custom_rand_crop_rotate(
                    [img_tensor],
                    d_size, hw_size, hw_size,
                    rotation_angles, rotation_methods,
                    ('bilinear',),
                    ensure_bothground=False)
                img_tensor = img_tensor[0]
        elif augmentation_method == 'Edge Replicate Pad' and random.random() < prob:
            img_tensor = Aug.edge_replicate_pad((img_tensor,), row['Value'])
        elif augmentation_method == 'Vertical Flip' and random.random() < prob:
            img_tensor = T_F.vflip(img_tensor)
        elif augmentation_method == 'Horizontal Flip' and random.random() < prob:
            img_tensor = T_F.hflip(img_tensor)
        elif augmentation_method == 'Depth Flip' and random.random() < prob:
            img_tensor = img_tensor.flip([1])
        elif augmentation_method == 'Simulate Low Resolution' and random.random() < prob:
            img_tensor = Aug.sim_low_res(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Noise' and random.random() < prob:
            img_tensor = Aug.gaussian_noise(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Blur' and random.random() < prob:
            img_tensor = Aug.gaussian_blur_3d(img_tensor, int(row['Value']),
                                              random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gradient Gamma' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'gamma')
        elif augmentation_method == 'Gradient Contrast' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'contrast')
        elif augmentation_method == 'Gradient Brightness' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), 'brightness')
        elif augmentation_method == 'Adjust Gamma' and random.random() < prob:
            img_tensor = Aug.adj_gamma(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Contrast' and random.random() < prob:
            img_tensor = Aug.adj_contrast(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Brightness' and random.random() < prob:
            img_tensor = Aug.adj_brightness(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Salt And Pepper' and random.random() < prob:
            img_tensor = Aug.salt_and_pepper_noise(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
    return img_tensor


def make_dataset_tv(image_dir, extensions=(".tif", ".tiff", ".hdf5")):
    """
    Generate a list containing pairs of file paths to training images and their labels.
    The labels should have the same name as their corresponding image, with a prefix "Labels_".

    Args:
        image_dir (str): Path to the directory where images are stored.
        extensions: I honestly don't know about this too, I guess it isolate image files?

    Returns:
        Example[('Datasets\\train\\testimg1.tif', 'Datasets\\train\\Labels_testimg1.tif'),
                ('Datasets\\train\\testimg2.tif', 'Datasets\\train\\Labels_testimg2.tif')]
    """
    image_label_pair = []
    image_files = os.listdir(image_dir)
    for fname in sorted(image_files):
        if has_file_allowed_extension(fname, extensions):
            if not "Labels_" in fname:
                path = os.path.join(image_dir, fname)
                label_path = os.path.join(image_dir, 'Labels_' + fname)
                image_label_pair.append((path, label_path))
    return image_label_pair


def make_dataset_predict(image_dir, extensions=(".tif", ".tiff", ".hdf5")):
    """
    Generate a list containing file paths to images waiting to get predicted.

    Args:
        image_dir (str): Path to the directory where images are stored.
        extensions: I honestly don't know about this too, I guess it isolate image files?

    Returns:
        Example['Datasets\\predict\\testpic1.tif',
                'Datasets\\predict\\testpic2.tif']
    """
    path_list = []
    image_dir = os.path.expanduser(image_dir)
    for root, _, fnames in sorted(os.walk(image_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.normpath(os.path.join(root, fname))
                path_list.append(path)
    return path_list


class TrainDataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the training dataset.

    Args:
        images_dir (str): Path to the directory where images are stored.
        augmentation_csv (str): Path to the .csv file that contains the image augmentations parameters.
        train_multiplier (int): A.k.a repeat, the number of training images in each epoch are multiplied by this number. Default: 1
        hw_size (int): The height and width of each generated patch. Default: 64
        d_size (int): The depth of each generated patch. Default: 64
        instance_mode (bool): If true, will prepare contour tensors for instance segmentation. Default: False.
        exclude_edge (bool): If true, the borders of the objects in the label tensor will be excluded from gradient calculation.
                             Will force to false if is instance segmentation mode.
                             Default: false
        exclude_edge_size_in (int): The thickness of the border in pixels, toward the inside of each object.
        exclude_edge_size_out (int): The thickness of the border in pixels, toward the outside of each object.
        contour_map_width (int): Width of the contour map. Default: 1.
        key (str): If trying to load from a hdf5 file, will load the File object with this name.
    """

    def __init__(self, images_dir, augmentation_csv, train_multiplier=1, hw_size=64, d_size=64, instance_mode=False,
                 exclude_edge=False, exclude_edge_size_in=1, exclude_edge_size_out=1, contour_map_width=1,
                 hdf5_key='Default'):
        # Get a list of file paths for images and labels
        self.file_list = np.array(make_dataset_tv(images_dir))
        self.num_files = len(self.file_list)
        self.instance_mode = instance_mode
        def place_holder(a): return a
        fc = place_holder
        if exclude_edge and not instance_mode:
            fc = Aug.exclude_border_labels
        if instance_mode:
            self.contour_tensors = [torch.from_numpy(get_contour_maps(item, 'generated_contour_maps', contour_map_width)) for item in self.file_list]
        self.lab_tensors = [torch.from_numpy(fc(Aug.binarisation(path_to_array(item[1], key=hdf5_key, label=True)))) for item in self.file_list]
        self.img_tensors = [torch.from_numpy(path_to_array(item[0], key=hdf5_key, label=False)) for item in self.file_list]
        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.train_multiplier = train_multiplier
        self.hw_size = hw_size
        self.d_size = d_size
        super().__init__()

    def __len__(self):
        return self.num_files * self.train_multiplier

    def __getitem__(self, idx):
        """
        negative_control (str or None): If 'negative', allow the returned lab_tensor to be fully negative (no foreground object).
                                        If 'positive', allow the returned lab_tensor to be fully positive. If None, will try to generate lab_tensor
                                        that contain both positive and negative pixels. Default: None
        """
        negative_control = idx[1]
        idx = idx[0]
        if negative_control:
            if negative_control == 'positive':
                foreground_threshold = 0.01
                background_threshold = 0
            elif negative_control == 'negative':
                background_threshold = 0.01
                foreground_threshold = 0
        else:
            background_threshold = 0.01
            foreground_threshold = 0.01
        idx = math.floor(idx / self.train_multiplier)
        img_tensor, lab_tensor = self.img_tensors[idx][None, :], self.lab_tensors[idx][None, :]
        if self.instance_mode:
            contour_tensor = self.contour_tensors[idx][None, :]
            img_tensor, lab_tensor, contour_tensor = apply_aug(img_tensor, lab_tensor, contour_tensor,
                                                               self.augmentation_params, self.hw_size, self.d_size,
                                                               foreground_threshold, background_threshold)
            return img_tensor, lab_tensor, contour_tensor
        else:
            img_tensor, lab_tensor = apply_aug(img_tensor, lab_tensor, None,
                                               self.augmentation_params, self.hw_size, self.d_size,
                                               foreground_threshold, background_threshold)
            return img_tensor, lab_tensor

    def get_label_mean(self):
        numels = 0
        sum = 0
        for array in self.lab_tensors:
            numels += array.dim()
            sum += array.sum()
        return sum / numels

    def get_contour_mean(self):
        numels = 0
        sum = 0
        for array in self.contour_tensors:
            numels += array.dim()
            sum += array.sum()
        return sum / numels


class UnsupervisedDataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the unsupervised training dataset.

    Args:
        images_dir (str): Path to the directory where images are stored.
        augmentation_csv (str): Path to the .csv file that contains the image augmentations parameters.
        train_multiplier (int): A.k.a repeat, the number of training images in each epoch are multiplied by this number. Default: 1
        hw_size (int): The height and width of each generated patch. Default: 64
        d_size (int): The depth of each generated patch. Default: 64
        key (str): If trying to load from a hdf5 file, will load the File object with this name.
    """
    def __init__(self, images_dir, augmentation_csv, train_multiplier=1, hw_size=64, d_size=64, hdf5_key='Default'):
        self.file_list = np.array(make_dataset_predict(images_dir))
        self.num_files = len(self.file_list)
        self.img_tensors = [torch.from_numpy(path_to_array(item, key=hdf5_key, label=False)) for item in self.file_list]
        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.train_multiplier = train_multiplier
        self.hw_size = hw_size
        self.d_size = d_size

    def __len__(self):
        return self.num_files * self.train_multiplier

    def __getitem__(self, idx):
        idx = math.floor(idx / self.train_multiplier)
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
        augmentation_csv (str): Path to the .csv file that contains the image augmentations parameters.
        contour_map_width (int): Width of the contour map. Default: 1.
        key (str): If trying to load from a hdf5 file, will load the File object with this name.
    """

    def __init__(self, images_dir, hw_size, d_size, instance_mode, augmentation_csv, contour_map_width=1, hdf5_key='Default'):
        # Get a list of file paths for images and labels
        file_list = np.array(make_dataset_tv(images_dir))
        self.num_files = len(file_list)
        self.instance_mode = instance_mode
        if instance_mode:
            tensor_pairs = [((path_to_array(item[0], key=hdf5_key, label=False)),
                             Aug.binarisation(path_to_array(item[1], label=True)),
                             get_contour_maps(item, 'generated_contour_maps', contour_map_width)) for item in file_list]
        else:
            tensor_pairs = [(path_to_array(item[0], key=hdf5_key, label=False),
                            Aug.binarisation(path_to_array(item[1], label=True))) for item in file_list]
        self.chopped_array_pairs = []
        # augmentation_params = pd.read_csv(augmentation_csv)
        # Crop the tensors, so they are the standard shape specified in the augmentation csv.
        for pairs in tensor_pairs:
            depth, height, width = pairs[0].shape
            depth_multiplier = math.ceil(depth / d_size)
            height_multiplier = math.ceil(height / hw_size)
            width_multiplier = math.ceil(width / hw_size)
            self.total_multiplier = depth_multiplier * height_multiplier * width_multiplier
            self.leave_out_list = []
            # Loop through each depth, height, and width index
            for depth_idx in range(depth_multiplier):
                for height_idx in range(height_multiplier):
                    for width_idx in range(width_multiplier):
                        # Calculate the start and end indices for depth, height, and width
                        if depth_multiplier > 1:
                            depth_start = (d_size - (
                                        (d_size * depth_multiplier - depth) / (depth_multiplier - 1))) * depth_idx
                            depth_start = math.floor(depth_start)
                        else:
                            depth_start = 0
                        depth_end = depth_start + d_size
                        if height_multiplier > 1:
                            height_start = (hw_size - ((hw_size * height_multiplier - height) / (
                                        height_multiplier - 1))) * height_idx
                            height_start = math.floor(height_start)
                        else:
                            height_start = 0
                        height_end = height_start + hw_size
                        if width_multiplier > 1:
                            width_start = (hw_size - (
                                        (hw_size * width_multiplier - width) / (width_multiplier - 1))) * width_idx
                            width_start = math.floor(width_start)
                        else:
                            width_start = 0
                        width_end = width_start + hw_size
                        cropped_img = torch.from_numpy(pairs[0][depth_start:depth_end, height_start:height_end,
                                                                width_start:width_end])
                        cropped_lab = torch.from_numpy(pairs[1][depth_start:depth_end, height_start:height_end,
                                                                width_start:width_end])
                        if instance_mode:
                            cropped_contour = torch.from_numpy(pairs[2][depth_start:depth_end,
                                                                        height_start:height_end,
                                                                        width_start:width_end])
                            self.chopped_array_pairs.append((cropped_img, cropped_lab, cropped_contour))
                        else:
                            self.chopped_array_pairs.append((cropped_img, cropped_lab))
        super().__init__()

    def __len__(self):
        return self.num_files * self.total_multiplier

    def __getitem__(self, idx):
        img_tensor, lab_tensor = self.chopped_array_pairs[idx][0][None, :], self.chopped_array_pairs[idx][1][None, :].to(torch.float32)
        if self.instance_mode:
            contour_tensor = self.chopped_array_pairs[idx][2][None, :].to(torch.float32)
            return img_tensor, lab_tensor, contour_tensor
        else:
            return img_tensor, lab_tensor


class Predict_Dataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the predict dataset.\n
    Note if the image is larger than the size specified in the augmentation_csv, it will get cropped into several
    (potentially) overlapping smaller images. The final results will be stitched together from predictions of these smaller images.\n
    The actual height and width of each patch is hw_size + 2 * hw_overlap. Same goes for depth.

    Args:
        images_dir (str): Path to the directory where images are stored.
        hw_size (int): The height and width patches the prediction image will be cropped to. In pixels.
        depth_size (int): The depth of patches the prediction image will be cropped to. In pixels.
        hw_overlap (int): The additional gain in height and width of the patches. In pixels. Helps smooth out borders between patches.
        depth_overlap (int): The additional gain in depth of the patches. In pixels. Helps smooth out borders between patches.
        hdf5_key (str): If trying to load from a hdf5 file, will load the File object with this name.
    """

    def __init__(self, images_dir, hw_size=128, depth_size=128, hw_overlap=16, depth_overlap=16,
                 leave_out_idx=None, hdf5_key='Default'):
        self.file_list = make_dataset_predict(images_dir)
        self.patches_list = []
        self.meta_list = []
        if leave_out_idx is not None:
            file = self.file_list[leave_out_idx]
            self.file_list = [file]
        for file in self.file_list:
            image = path_to_array(file, key=hdf5_key, label=False)[None, :]
            shape = image.shape
            depth = shape[1]
            height = shape[2]
            width = shape[3]
            file_name = os.path.basename(file)

            # Calculate the multipliers for padding and cropping
            depth_multiplier = math.ceil(depth / depth_size)
            height_multiplier = math.ceil(height / hw_size)
            width_multiplier = math.ceil(width / hw_size)
            # Padding and cropping
            paddings = ((0, 0),
                        (depth_overlap, depth_overlap),
                        (hw_overlap, hw_overlap),
                        (hw_overlap, hw_overlap))
            image = np.pad(image, paddings, mode="symmetric")
            # Loop through each depth, height, and width index
            for depth_idx in range(depth_multiplier):
                for height_idx in range(height_multiplier):
                    for width_idx in range(width_multiplier):
                        if depth_multiplier > 1:
                            depth_start = (depth_size - ((depth_size * depth_multiplier - depth) / (
                                        depth_multiplier - 1))) * depth_idx
                            depth_start = math.floor(depth_start)
                        else:
                            depth_start = 0
                        depth_end = depth_start + depth_size + (2 * depth_overlap)
                        if height_multiplier > 1:
                            height_start = (hw_size - ((hw_size * height_multiplier - height) / (
                                        height_multiplier - 1))) * height_idx
                            height_start = math.floor(height_start)
                        else:
                            height_start = 0
                        height_end = height_start + hw_size + (2 * hw_overlap)
                        if width_multiplier > 1:
                            width_start = (hw_size - (
                                        (hw_size * width_multiplier - width) / (width_multiplier - 1))) * width_idx
                            width_start = math.floor(width_start)
                        else:
                            width_start = 0
                        width_end = width_start + hw_size + (2 * hw_overlap)
                        patch = image[:,
                                      depth_start:depth_end,
                                      height_start:height_end,
                                      width_start:width_end]
                        self.patches_list.append(torch.from_numpy(patch))
            self.meta_list.append((file_name, shape))
        super().__init__()

    def __len__(self):
        return len(self.patches_list)

    def __getitem__(self, idx):
        return self.patches_list[idx]

    def __getmetainfo__(self):
        return self.meta_list


class CollectedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_supervised, dataset_unsupervised=None):
        self.dataset_supervised = dataset_supervised
        self.dataset_unsupervised = dataset_unsupervised
        if self.dataset_unsupervised is not None:
            self.rectified_unsupervised_size = math.floor(len(self.dataset_unsupervised) / 2) * 2

    def __len__(self):
        if self.dataset_unsupervised:
            return len(self.dataset_supervised) + self.rectified_unsupervised_size
        else:
            return len(self.dataset_supervised)

    def __getitem__(self, idx):
        if self.dataset_unsupervised:
            if idx[0] < self.rectified_unsupervised_size:
                return self.dataset_unsupervised[idx[0]]
            else:
                idx[0] = idx[0] - self.rectified_unsupervised_size
                return self.dataset_supervised[idx]
        else:
            return self.dataset_supervised[idx]


class CollectedSampler(torch.utils.data.Sampler):
    def __init__(self, data, batch_size, dataset_unsupervised=None):
        super(CollectedSampler, self).__init__(data)
        self.data = data
        self.batch_size = batch_size
        if dataset_unsupervised:
            self.dataset_unsupervised_size = math.floor(len(dataset_unsupervised) / 2) * 2
        else:
            self.dataset_unsupervised_size = 0

    def __iter__(self):
        if self.batch_size == 1:
            if self.dataset_unsupervised_size == 0:
                array = np.random.permutation(len(self.data))
                array = [[num, None] for num in array] # To indicate that no negative control is used.
                return iter(array)
            else:
                unsupervised_array = np.random.permutation(self.dataset_unsupervised_size)
                unsupervised_array = [[num, 'unsupervised'] for num in unsupervised_array]
                supervised_array = np.arange(self.dataset_unsupervised_size, len(self.data))
                np.random.shuffle(supervised_array)
                supervised_array = [[num, None] for num in supervised_array]
                min_size = min(self.dataset_unsupervised_size, len(supervised_array))
                interleaved_array = []
                for i in range(0, min_size):
                    interleaved_array.append(supervised_array[i])
                    interleaved_array.append(unsupervised_array[i])
                if len(supervised_array) > min_size:
                    interleaved_array.extend(supervised_array[min_size:])
                elif self.dataset_unsupervised_size > min_size:
                    interleaved_array.extend(unsupervised_array[min_size:])
                return iter(interleaved_array)
        elif self.batch_size == 2:
            if self.dataset_unsupervised_size == 0:
                original_array = np.arange(len(self.data))
                positive_array, negative_array = np.split(original_array, 2)
                np.random.shuffle(positive_array)
                positive_array = [[num, 'positive'] for num in positive_array]
                np.random.shuffle(negative_array)
                negative_array = [[num, 'negative'] for num in negative_array]
                shuffled_array = []
                for i in range(0, len(positive_array)):
                    shuffled_array.append(positive_array[i])
                    shuffled_array.append(negative_array[i])
                return iter(shuffled_array)
            else:
                unsupervised_array = np.arange(self.dataset_unsupervised_size)
                np.random.shuffle(unsupervised_array)
                unsupervised_array = [[num, 'unsupervised'] for num in unsupervised_array]
                supervised_array = np.arange(self.dataset_unsupervised_size, len(self.data))
                positive_array, negative_array = np.split(supervised_array, 2)
                np.random.shuffle(positive_array)
                positive_array = [[num, 'positive'] for num in positive_array]
                np.random.shuffle(negative_array)
                negative_array = [[num, 'negative'] for num in negative_array]
                supervised_array = []
                for i in range(0, len(positive_array)):
                    supervised_array.append(positive_array[i])
                    supervised_array.append(negative_array[i])

                min_size = min(self.dataset_unsupervised_size, len(supervised_array))

                # Interleave the pairs as much as possible
                interleaved_array = []
                for i in range(0, min_size, 2):
                    interleaved_array.append(supervised_array[i])  # positive
                    interleaved_array.append(supervised_array[i + 1])  # negative
                    interleaved_array.append(unsupervised_array[i])
                    interleaved_array.append(unsupervised_array[i + 1])

                # Append the remaining pairs from the larger dataset
                if len(supervised_array) > min_size:
                    interleaved_array.extend(supervised_array[min_size:])
                elif self.dataset_unsupervised_size > min_size:
                    interleaved_array.extend(unsupervised_array[min_size:])

                return iter(interleaved_array)
        else:
            raise ValueError("Invalid batch size. We only support 1 or 2.")

    def __len__(self):
        return len(self.data)


def custom_collate(batch):
    if len(batch) == 1:
        batch = [torch.unsqueeze(sample, dim=0) for sample in batch[0]]
        return batch
    positive_samples = batch[0]
    negative_samples = batch[1]
    batch = [torch.stack((a_i, b_i), dim=0) for a_i, b_i in zip(positive_samples, negative_samples)]
    return batch


def get_contour_maps(file_name, folder_path, contour_map_width):
    """
    Try get the contour map for the images file. Will first try to load previously saved one from folder_path,
    if failed, will generate new one and save it to folder_path.
    """
    img_path = file_name[1]
    dir_name, img_name = os.path.split(img_path)
    contour_img_name = "contour_" + img_name
    contour_img_path = os.path.join(folder_path, contour_img_name)
    if not os.path.exists(contour_img_path):
        print(f'Generating contour map for {img_name}... Can take a while if there are lots of objects.')
        contour_img = Aug.instance_contour_transform(path_to_array(img_path, label=True), contour_outward=contour_map_width)
        imageio.v3.imwrite(uri=f'{contour_img_path}', image=contour_img)
        print(f'Saved {contour_img_name}')
    else:
        contour_img = path_to_array(contour_img_path, label=True)
        print(
            f'Loaded previously saved {contour_img_name}. Remember to delete old one if you made change to the label!')
    return contour_img


class CrossValidationDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, augmentation_csv, hw_size, d_size, leave_out_index=0, mode='Train', train_multiplier=1):
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        self.leave_out = self.file_list.pop(leave_out_index)
        self.img_tensors = [path_to_array(item[0], label=False) for item in self.file_list]
        self.lab_tensors = [path_to_array(item[1], label=True) for item in self.file_list]
        self.train_multiplier = train_multiplier

        self.leave_out_img = path_to_array(self.leave_out[0], label=False)[None, :]
        leave_out_img_name = os.path.basename(self.leave_out[0])
        self.leave_out_label = path_to_array(self.leave_out[1], label=True)[None, :]
        self.mode = mode
        self.augmentation_params = pd.read_csv(augmentation_csv)

        unused, self.depth, self.height, self.width = self.leave_out_img.shape
        depth_multiplier = math.ceil(self.depth / d_size)
        height_multiplier = math.ceil(self.height / hw_size)
        width_multiplier = math.ceil(self.width / hw_size)
        self.total_multiplier = depth_multiplier * height_multiplier * width_multiplier
        self.leave_out_list = []
        self.meta_list = []
        # Loop through each depth, height, and width index
        for depth_idx in range(depth_multiplier):
            for height_idx in range(height_multiplier):
                for width_idx in range(width_multiplier):
                    # Calculate the start and end indices for depth, height, and width
                    if depth_multiplier > 1:
                        depth_start = math.floor(
                            (d_size - ((d_size * depth_multiplier - self.depth) / (depth_multiplier - 1))) * depth_idx)
                    else:
                        depth_start = 0
                    depth_end = depth_start + hw_size
                    if height_multiplier > 1:
                        height_start = math.floor((hw_size - (
                                    (hw_size * height_multiplier - self.height) / (height_multiplier - 1))) * height_idx)
                    else:
                        height_start = 0
                    height_end = height_start + hw_size
                    if width_multiplier > 1:
                        width_start = math.floor(
                            (hw_size - ((hw_size * width_multiplier - self.width) / (width_multiplier - 1))) * width_idx)
                    else:
                        width_start = 0
                    width_end = width_start + hw_size
                    cropped_img = self.leave_out_img[:, depth_start:depth_end, height_start:height_end,
                                  width_start:width_end]
                    cropped_lab = self.leave_out_label[:, depth_start:depth_end, height_start:height_end,
                                  width_start:width_end]
                    self.leave_out_list.append((cropped_img, cropped_lab))
        self.meta_list.append((leave_out_img_name, self.leave_out_img.shape))
        super().__init__()

    def __len__(self):
        if self.mode == 'Train':
            return (self.num_files - 1) * self.train_multiplier
        else:
            return self.total_multiplier

    def __getitem__(self, idx):
        if self.mode == 'Train':
            idx = math.floor(idx / self.train_multiplier)
            img_tensor, lab_tensor = self.img_tensors[idx][None, :], self.lab_tensors[idx][None, :]
            img_tensor, lab_tensor = apply_aug(img_tensor, lab_tensor, self.augmentation_params)
            return img_tensor, lab_tensor
        elif self.mode == 'Val':
            img_tensor, lab_tensor = self.leave_out_list[idx][0], self.leave_out_list[idx][1]
            return img_tensor, lab_tensor
        else:
            img_tensor = self.leave_out_list[idx][0][None, :]
            return img_tensor

    def __getmetainfo__(self):
        return self.meta_list


def stitch_output_volumes(tensor_list, meta_list, hw_size=128, depth_size=128, hw_overlap=16, depth_overlap=16):
    """
    Stitch the patches of output volumes and reconstruct the original image tensor(s).

    Args:
        tensor_list (list): List of image tensors with shape (C, D, H, W).
        meta_list (list): The meta information auto generated by Predict_Dataset class.
        hw_size (int): The height and width of the patches. In pixels.
        depth_size (int): The depth of the patches. In pixels.
        hw_overlap (int): The additional gain in height and width of the patches. In pixels.
        depth_overlap (int): The additional gain in depth of the patches. In pixels.

    Returns:
        list: stitched tensors with shape (C, D, H, W).
    """
    output_list = []

    for meta_info in meta_list:
        depth, height, width = meta_info[1][1:]
        file_name = "Mask_" + meta_info[0]

        depth_multiplier = math.ceil(depth / depth_size)
        height_multiplier = math.ceil(height / hw_size)
        width_multiplier = math.ceil(width / hw_size)
        total_multiplier = depth_multiplier * height_multiplier * width_multiplier

        # f32 would be a waste here
        result_volume = torch.zeros((depth,
                                     height,
                                     width), dtype=torch.float16)

        for i in range(total_multiplier):
            tensor_work_with = tensor_list[i]
            if hw_overlap or depth_overlap:
                tensor_work_with = tensor_work_with[
                                   (depth_overlap if depth_overlap else slice(None)):
                                   -(depth_overlap if depth_overlap else slice(None)),
                                   (hw_overlap if hw_overlap else slice(None)):
                                   -(hw_overlap if hw_overlap else slice(None)),
                                   (hw_overlap if hw_overlap else slice(None)):
                                   -(hw_overlap if hw_overlap else slice(None))
                                   ]

            depth_idx, height_idx, width_idx = (i // (height_multiplier * width_multiplier)) % depth_multiplier, \
                                               (i // width_multiplier) % height_multiplier, \
                                               i % width_multiplier

            depth_start = math.floor((depth_size - (depth_size * depth_multiplier - depth) / (
                        depth_multiplier - 1)) * depth_idx) if depth_multiplier > 1 else 0
            height_start = math.floor((hw_size - (hw_size * height_multiplier - height) / (
                        height_multiplier - 1)) * height_idx) if height_multiplier > 1 else 0
            width_start = math.floor((hw_size - (hw_size * width_multiplier - width) / (
                        width_multiplier - 1)) * width_idx) if width_multiplier > 1 else 0

            result_volume[depth_start:depth_start + tensor_work_with.shape[0],
                          height_start:height_start + tensor_work_with.shape[1],
                          width_start:width_start + tensor_work_with.shape[2]] = tensor_work_with

        output_list.append((result_volume, file_name))

    return output_list


def predictions_to_final_img(predictions, meta_list, direc, hw_size=128, depth_size=128, hw_overlap=16,
                             depth_overlap=16):
    """
    Stitch the patches of prediction output from network and save it in the selected directory.\n
    This one is for semantic segmentation.

    Args:
        predictions (list): Output from trainer.predict()
        meta_list (list): The meta information auto generated by Predict_Dataset class.
        direc (str): Directory where the final images will be saved to.
        hw_size (int): The height and width of the patches. In pixels.
        depth_size (int): The depth of the patches. In pixels.
        hw_overlap (int): The additional gain in height and width of the patches. In pixels.
        depth_overlap (int): The additional gain in depth of the patches. In pixels.
    """
    tensor_list = [torch.squeeze(tensor, (0, 1))
                   for prediction in predictions
                   for tensor in torch.split(prediction[0], 1, dim=0)]

    stitched_volumes = stitch_output_volumes(tensor_list, meta_list, hw_size, depth_size, hw_overlap, depth_overlap)
    del tensor_list
    for volume in stitched_volumes:
        array = volume[0].numpy()
        array = np.where(array >= 0.5, np.uint8(1), np.uint8(0))
        imageio.v3.imwrite(uri=f'{direc}/{volume[1]}', image=array)


def predictions_to_final_img_instance(predictions, meta_list, direc, hw_size=128, depth_size=128, hw_overlap=16,
                                      depth_overlap=16, segmentation_mode='simple', dynamic=10, pixel_reclaim=True):
    """
    Stitch the patches of prediction output from network and save it in the selected directory.\n
    This one is for instance segmentation.

    Args:
        predictions (list): Output from trainer.predict()
        meta_list (list): The meta information auto generated by Predict_Dataset class.
        direc (str): Directory where the final images will be saved to.
        hw_size (int): The height and width of the patches. In pixels.
        depth_size (int): The depth of the patches. In pixels.
        hw_overlap (int): The additional gain in height and width of the patches. In pixels.
        depth_overlap (int): The additional gain in depth of the patches. In pixels.
        segmentation_mode (str): If 'simple', will identify objects via simple connected component labelling. If 'watershed', will use a distance transform watershed instead, which is slower but yield much less under-segment.
        dynamic (int): Dynamic of intensity for the search of regional minima in the distance transform image. Increasing its value will yield more object merges. Default: 10.
        pixel_reclaim (bool): Whether to reclaim lost pixel during the instance segmentation, a slow process. Default: True
    """
    tensor_list_p = [
        torch.squeeze(tensor, (0, 1))
        for prediction in predictions
        for tensor in torch.split(prediction[0], 1, dim=0)
    ]

    tensor_list_c = [
        torch.squeeze(tensor, (0, 1))
        for prediction in predictions
        for tensor in torch.split(prediction[1], 1, dim=0)
    ]

    del predictions
    stitched_volumes_p = stitch_output_volumes(tensor_list_p, meta_list, hw_size, depth_size, hw_overlap, depth_overlap)
    del tensor_list_p
    stitched_volumes_c = stitch_output_volumes(tensor_list_c, meta_list, hw_size, depth_size, hw_overlap, depth_overlap)
    del tensor_list_c
    #    for volume in stitched_volumes_p:
    #        array = np.asarray(volume[0])
    #        imageio.v3.imwrite(uri=f'{direc}/Pixels_{volume[1]}', image=np.uint8(array))
    #    for volume in stitched_volumes_c:
    #        array = np.asarray(volume[0])
    #        imageio.v3.imwrite(uri=f'{direc}/Contour_{volume[1]}', image=np.float32(array))
    for semantic, contour in zip(stitched_volumes_p, stitched_volumes_c):
        imageio.v3.imwrite(uri=f'{direc}/Pixels_{semantic[1]}', image=np.float16(semantic[0].numpy()))
        imageio.v3.imwrite(uri=f'{direc}/Contour_{contour[1]}', image=np.float16(contour[0].numpy()))
        print(f'Computing instance segmentation using contour data for {contour[1]}... Can take a while if the image is big.')
        instance_array = instance_segmentation_simple(semantic[0], contour[0], mode=segmentation_mode, dynamic=dynamic, pixel_reclaim=pixel_reclaim)
        imageio.v3.imwrite(uri=f'{direc}/Instance_{contour[1]}', image=instance_array)


#    for volume in instance_result:
#        array = np.asarray(volume)
#        imageio.v3.imwrite(uri=f'{direc}/Instance_1', image=np.uint8(array))


'''
def hovernet_map_transform(input_tensor):
    # Identify unique objects in the tensor
    unique_values = torch.unique(input_tensor)

    # Create new tensors to store the result
    result_tensor_h = torch.zeros_like(input_tensor, dtype=torch.float32)
    result_tensor_v = torch.zeros_like(input_tensor, dtype=torch.float32)
    result_tensor_d = torch.zeros_like(input_tensor, dtype=torch.float32)

    for value in unique_values:
        if value == 0:
            continue  # Skip background

        # Create a binary mask for the current object
        object_mask = (input_tensor == value).float()

        if (object_mask > 0).sum().item() <= 10:
            continue  # Sanity Check

        # Calculate center of mass
        mass_center = torch.stack(torch.where(object_mask)).float().mean(dim=1)

        # Calculate distances from the center of mass
        horizontal_distance = torch.arange(object_mask.shape[2], dtype=torch.float32) - mass_center[2]
        vertical_distance = torch.arange(object_mask.shape[1], dtype=torch.float32) - mass_center[1]
        depth_distance = torch.arange(object_mask.shape[0], dtype=torch.float32) - mass_center[0]

        h_patch = object_mask * horizontal_distance.unsqueeze(0).unsqueeze(0)
        h_patch = h_patch / torch.max(torch.abs(h_patch)) if torch.max(torch.abs(h_patch)) > 0 else h_patch
        v_patch = object_mask * vertical_distance.unsqueeze(1).unsqueeze(0)
        v_patch = v_patch / torch.max(torch.abs(v_patch)) if torch.max(torch.abs(v_patch)) > 0 else v_patch
        d_patch = object_mask * depth_distance.unsqueeze(1).unsqueeze(1)
        d_patch = d_patch / torch.max(torch.abs(d_patch)) if torch.max(torch.abs(d_patch)) > 0 else d_patch

        # Assign new values based on distances
        result_tensor_h += h_patch
        result_tensor_v += v_patch
        result_tensor_d += d_patch

    output = torch.stack((result_tensor_d, result_tensor_v, result_tensor_h), 0)
    return output
'''

# Global variables to hold the shared memory objects in workers
_segmentation_shm = None
_touching_shm = None

def pool_initializer(shm_segmentation_name,
                     shm_touching_name, shm_touching_shape, shm_touching_dtype):
    global _segmentation_shm, _touching_shm, _touching_shape, _touching_dtype
    _segmentation_shm = shared_memory.SharedMemory(name=shm_segmentation_name)
    _touching_shm = shared_memory.SharedMemory(name=shm_touching_name)
    _touching_shape = shm_touching_shape
    _touching_dtype = shm_touching_dtype


def allocate_pixels_global(batch_indices_and_args):
    # Unpack batch indices and other args
    batch_indices, map_size, distance_threshold, minlength = batch_indices_and_args
    # Instead of re-opening the shared memories, use the global variables.
    touching_pixels = np.ndarray(_touching_shape, dtype=_touching_dtype, buffer=_touching_shm.buf)
    segmentation_shared = np.ndarray(map_size, dtype=np.uint16, buffer=_segmentation_shm.buf)

    for pixel_idx in batch_indices:
        z, y, x = touching_pixels[pixel_idx, 0], touching_pixels[pixel_idx, 1], touching_pixels[pixel_idx, 2]
        local_segment = segmentation_shared[
                        max(z - distance_threshold, 0):min(z + distance_threshold, map_size[0]),
                        max(y - distance_threshold, 0):min(y + distance_threshold, map_size[1]),
                        max(x - distance_threshold, 0):min(x + distance_threshold, map_size[2])
                        ]
        object_counts = np.bincount(local_segment.flatten(), minlength=minlength)
        object_counts = object_counts[1:]
        if object_counts.sum() > 0:
            closest_object = np.argmax(object_counts) + 1
            segmentation_shared[z, y, x] = closest_object.item()


def instance_segmentation_simple(semantic_map, contour_map, size_threshold=10, mode='simple', dynamic=10, pixel_reclaim=True, distance_threshold=1, batch_size=2048):
    """
    Using a semantic segmentation map and a contour map to separate touching objects and perform instance segmentation.
    Pixels in touching areas are assigned to the closest object based on the largest proportion of pixels within 5 pixel distance to the pixel.

    Args:
        semantic_map (torch.Tensor): The input semantic segmented map.
        contour_map (torch.Tensor): The input contour segmented map.
        size_threshold (int): The minimal size in pixel of each object. Object smaller than this will be removed.
        mode (str): If 'simple', will identify objects via simple connected component labelling. If 'watershed', will use a distance transform watershed instead, which is slower but yield much less under-segment.
        dynamic (int): Dynamic of intensity for the search of regional minima in the distance transform image. Increasing its value will yield more object merges. Default: 10.
        pixel_reclaim (bool): Whether to reclaim lost pixel during the instance segmentation, a slow process. Default: True
        distance_threshold (int): The radius in pixels to search for nearby pixels when allocating. Default: 1.
        batch_size (int): Batch size for pixel reclaim. Default: 2048.

    Returns:
        np.Array: uint16 instance segmented map where 0 is background and every other value represent an object.
    """
    semantic_map = torch.where(semantic_map >= 0.5, True, False)
    # Treat contour map with lower threshold to ensure separation
    contour_map = torch.where(contour_map >= 0.2, True, False)

    # Find boundary that lies within foreground objects
    touching_map = torch.logical_and(contour_map, semantic_map)
    # Remove touching area between foreground objects
    segmentation = torch.logical_xor(semantic_map, touching_map)
    del semantic_map, contour_map
    segmentation = segmentation.numpy().astype(np.uint16, copy=False)
    gc.collect()

    structure = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    ], dtype=np.byte)
    if mode == 'simple':
        label(segmentation, structure=structure, output=segmentation)
    elif mode == 'watershed':
        distance_map = Morph.chamferdistancetransform3duint16(segmentation)
        distance_map = Morph.inverter(distance_map)
        marker = distance_map + dynamic
        hmin = Morph.geodesicreconstructionbyerosion3d(marker, distance_map)
        del marker
        gc.collect()
        hmin = morph.local_minima(hmin).astype(np.uint16)
        label(hmin, output=hmin)
        print("Starts watershed flooding...")
        segmentation = Morph.watershed_3d(distance_map, markers=hmin, mask=segmentation)
        del distance_map, hmin
        gc.collect()
    morph.remove_small_objects(segmentation, min_size=size_threshold, connectivity=structure, out=segmentation)

    del structure

    if pixel_reclaim:
        touching_pixels = torch.nonzero(touching_map, as_tuple=False).to(torch.uint16).numpy()
        del touching_map
        num_touching_pixels = len(touching_pixels)
        minlength = segmentation.max().item()
        map_size = segmentation.shape

        # Create shared memory for segmentation
        shm_segmentation = shared_memory.SharedMemory(create=True,
                                                      size=segmentation.size * np.dtype(np.uint16).itemsize)
        segmentation_shared = np.ndarray(map_size, dtype=np.uint16, buffer=shm_segmentation.buf)
        segmentation_shared[:] = segmentation
        del segmentation  # Free memory
        gc.collect()

        # Create shared memory for touching_pixels
        shm_touching = shared_memory.SharedMemory(create=True, size=touching_pixels.nbytes)
        touching_pixels_shared = np.ndarray(touching_pixels.shape, dtype=touching_pixels.dtype,
                                            buffer=shm_touching.buf)
        touching_pixels_shared[:] = touching_pixels[:]
        del touching_pixels  # Free memory
        gc.collect()

        batch_indices = [list(range(i, min(i + batch_size, num_touching_pixels)))
                         for i in range(0, num_touching_pixels, batch_size)]

        # Prepare a simplified list of arguments that each worker needs
        worker_args = [
            (batch, map_size, distance_threshold, minlength)
            for batch in batch_indices
        ]

        start_time = time.time()
        # Create the pool with an initializer that attaches the shared memories
        with Pool(cpu_count(), initializer=pool_initializer,
                  initargs=(shm_segmentation.name,
                            shm_touching.name, touching_pixels_shared.shape, touching_pixels_shared.dtype)) as pool:
            pool.map(allocate_pixels_global, worker_args)
        elapsed_time = time.time() - start_time
        print(f"Time taken for pixel reclaim: {elapsed_time}")
        # Convert shared memory to tensor
        segmentation_result = np.copy(segmentation_shared)
        # Clean up shared memory
        shm_segmentation.close()
        shm_segmentation.unlink()
        shm_touching.close()
        shm_touching.unlink()
        img_max = segmentation_result.max()
        if len(np.unique(segmentation_result)) <= 2:
            new_dtype = np.bool_
        elif img_max <= np.iinfo(np.uint8).max:
            new_dtype = np.uint8
        elif img_max <= np.iinfo(np.uint16).max:
            new_dtype = np.uint16
        else:
            new_dtype = np.uint32
        segmentation_result = segmentation_result.astype(new_dtype, copy=False)
        return segmentation_result
    else:
        img_max = segmentation.max()
        if len(np.unique(segmentation)) <= 2:
            new_dtype = np.bool_
        elif img_max <= np.iinfo(np.uint8).max:
            new_dtype = np.uint8
        elif img_max <= np.iinfo(np.uint16).max:
            new_dtype = np.uint16
        else:
            new_dtype = np.uint32
        segmentation = segmentation.astype(new_dtype, copy=False)
        return segmentation


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    semantic = torch.tensor(imageio.v3.imread('Datasets/result/Pixels_Mask_bin2_854_20C_IA_1_189.tif'))
    contour = torch.tensor(imageio.v3.imread('Datasets/result/Contour_Mask_bin2_854_20C_IA_1_189.tif'))
    instance = instance_segmentation_simple(semantic, contour)
    imageio.v3.imwrite(uri='Datasets/result/Instance_Mask_bin2_854_20C_IA_1_189.tif', image=np.uint16(instance))