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
# import scipy
import pandas as pd
from skimage import morphology
from scipy.ndimage import label
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS
from . import Augmentations as Aug

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_label_fname(fname):
    return 'Labels_' + fname


# 输入图像或者标签的路径，得到已标准化的图像张量或者标签的张量
def path_to_tensor(path, label=False, auto_normalise=True):
    """
    Transform a path to an image file into a Pytorch tensor.

    Args:
        path (str): Path to the image file.
        label (bool): If false, the output will be a float32 tensor range from 0 to 1. Default: False.
        auto_normalise (bool): If true and label is false, the output will be normalised to ensure the maximal and minimal value in the tensor are 1 and 0.
                               Else it would just be divided by the maximum value allowed for its datatype, to ensure the new maximum value won't surpass 1.
                               Default: True.

    Returns:
        torch.Tensor: transformed Tensor.
    """
    # ToTensor()对16位图不方便，因此才用这招
    img = imageio.v3.imread(path)
    if label:
        if img.dtype == np.uint16 or img.dtype == np.uint32:
            img = img.astype(np.int32)
    else:
        if auto_normalise:
            # Calculate the low and high threshold values
            zero_point_one_percent_low = np.percentile(img, 0.1)
            zero_point_one_percent_high = np.percentile(img, 99.9)

            # Clip the values to be within the specified range
            img = np.clip(img, zero_point_one_percent_low, zero_point_one_percent_high)

            # Normalize to the range [0, 1]
            img = (img - zero_point_one_percent_low) / (zero_point_one_percent_high - zero_point_one_percent_low)
            img = img.astype(np.float32)
        else:
            max_value = np.iinfo(img.dtype).max if img.dtype.kind == 'u' else np.finfo(img.dtype).max
            img = (img / max_value).astype(np.float32)
    return torch.from_numpy(img)


def apply_aug(img_tensor, lab_tensor, contour_tensor, augmentation_params, hw_size, d_size, foreground_threshold):
    """
    Apply Image Augmentations to an image tensor and its label tensor using the augmentation parameters from a DataFrame.
    Can have an optional contour tensor processed as well.

    Args:
        img_tensor (torch.Tensor): Image tensor. Should be float32.
        lab_tensor (torch.Tensor): Label tensor, should be the same shape as img_tensor.
        contour_tensor (torch.Tensor or None): Optional Contour tensor, should be the same shape as img_tensor.
        augmentation_params (DataFrame): The DataFrame which the augmentation parameters will be used from.
        hw_size (int): The height and width of each generated patch.
        d_size (int): The depth of each generated patch.
        foreground_threshold (float): Proportion of desired minimal foreground pixels in the produced label tensor.

    Returns:
        Transformed Image and Label Tensor.
    """
    for _, row in augmentation_params.iterrows():
        augmentation_method, prob = row['Augmentation'], row['Probability']
        if augmentation_method == 'Rescaling':
            if random.random() < prob:
                scale = random.uniform(row['Low Bound'], row['High Bound'])
                if contour_tensor is not None:
                    img_tensor, lab_tensor, contour_tensor = Aug.custom_rand_crop(
                        [img_tensor, lab_tensor, contour_tensor],
                        int(scale * d_size),
                        int(scale * hw_size),
                        int(scale * hw_size),
                        minimal_foreground=foreground_threshold)
                    contour_tensor = contour_tensor[None, :]
                    contour_tensor = F.interpolate(contour_tensor, size=(d_size, hw_size, hw_size),
                                                   mode="nearest-exact")
                    contour_tensor = torch.squeeze(contour_tensor, 0)
                else:
                    img_tensor, lab_tensor = Aug.custom_rand_crop([img_tensor, lab_tensor],
                                                                  int(scale * d_size),
                                                                  int(scale * hw_size),
                                                                  int(scale * hw_size),
                                                                  minimal_foreground=foreground_threshold)
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
                    img_tensor, lab_tensor, contour_tensor = Aug.custom_rand_crop(
                        [img_tensor, lab_tensor, contour_tensor],
                        d_size, hw_size, hw_size,
                        minimal_foreground=foreground_threshold)
                else:
                    img_tensor, lab_tensor = Aug.custom_rand_crop(
                        [img_tensor, lab_tensor],
                        d_size, hw_size, hw_size,
                        minimal_foreground=foreground_threshold)
        elif augmentation_method == 'Rotate xy' and random.random() < prob:
            if contour_tensor is not None:
                img_tensor, lab_tensor, contour_tensor = Aug.random_rotation_3d([img_tensor, lab_tensor, contour_tensor],
                                                                                interpolations=('bilinear', 'nearest', 'nearest'),
                                                                                fill_values=(0, 0, 0),
                                                                                plane='xy', angle_range=(row['Low Bound'], row['High Bound']))
            else:
                img_tensor, lab_tensor = Aug.random_rotation_3d([img_tensor, lab_tensor],
                                                                interpolations=('bilinear', 'nearest'),
                                                                plane='xy',
                                                                angle_range=(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate xz' and random.random() < prob:
            if contour_tensor is not None:
                img_tensor, lab_tensor, contour_tensor = Aug.random_rotation_3d([img_tensor, lab_tensor, contour_tensor],
                                                                                interpolations=('bilinear', 'nearest', 'nearest'),
                                                                                fill_values=(0, 0, 0),
                                                                                plane='xz', angle_range=(row['Low Bound'], row['High Bound']))
            else:
                img_tensor, lab_tensor = Aug.random_rotation_3d([img_tensor, lab_tensor],
                                                                interpolations=('bilinear', 'nearest'),
                                                                plane='xz',
                                                                angle_range=(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate yz' and random.random() < prob:
            if contour_tensor is not None:
                img_tensor, lab_tensor, contour_tensor = Aug.random_rotation_3d([img_tensor, lab_tensor, contour_tensor],
                                                                                interpolations=('bilinear', 'nearest', 'nearest'),
                                                                                fill_values=(0, 0, 0),
                                                                                plane='yz', angle_range=(row['Low Bound'], row['High Bound']))
            else:
                img_tensor, lab_tensor = Aug.random_rotation_3d([img_tensor, lab_tensor],
                                                                interpolations=('bilinear', 'nearest'),
                                                                plane='yz',
                                                                angle_range=(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Vertical Flip' and random.random() < prob:
            img_tensor, lab_tensor = T_F.vflip(img_tensor), T_F.vflip(lab_tensor)
            if contour_tensor is not None:
                contour_tensor = T_F.vflip(contour_tensor)
        elif augmentation_method == 'Horizontal Flip' and random.random() < prob:
            img_tensor, lab_tensor = T_F.hflip(img_tensor), T_F.hflip(lab_tensor)
            if contour_tensor is not None:
                contour_tensor = T_F.hflip(contour_tensor)
        elif augmentation_method == 'Simulate Low Resolution' and random.random() < prob:
            img_tensor = Aug.sim_low_res(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Blur' and random.random() < prob:
            img_tensor = Aug.gaussian_blur_3d(img_tensor, int(row['Value']),
                                              random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gradient Gamma' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), True)
        elif augmentation_method == 'Gradient Contrast' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']), False)
        elif augmentation_method == 'Adjust Contrast' and random.random() < prob:
            img_tensor = Aug.adj_contrast(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Gamma' and random.random() < prob:
            img_tensor = Aug.adj_gamma(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Brightness' and random.random() < prob:
            img_tensor = Aug.adj_brightness(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Salt And Pepper' and random.random() < prob:
            img_tensor = Aug.salt_and_pepper_noise(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Label Blur' and random.random() < prob:
            lab_tensor = lab_tensor.to(torch.float32)
            lab_tensor = Aug.gaussian_blur_3d(lab_tensor, int(row['Value']),
                                              random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Contour Blur' and random.random() < prob and contour_tensor is not None:
            contour_tensor = contour_tensor.to(torch.float32)
            contour_tensor = Aug.gaussian_blur_3d(contour_tensor, int(row['Value']),
                                                  random.uniform(row['Low Bound'], row['High Bound']))
    if contour_tensor is not None:
        return img_tensor, lab_tensor, contour_tensor
    else:
        return img_tensor, lab_tensor


def make_dataset_tv(image_dir, extensions=IMG_EXTENSIONS):
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


def make_dataset_predict(image_dir, extensions=IMG_EXTENSIONS):
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
        allow_negative (bool): If true, allow the returned lab_tensor to be fully negative (no foreground object). Default: false
    """

    def __init__(self, images_dir, augmentation_csv, train_multiplier=1, hw_size=64, d_size=64, instance_mode=False,
                 exclude_edge=False, exclude_edge_size_in=1, exclude_edge_size_out=1, contour_map_width=1, allow_negative=False):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        self.instance_mode = instance_mode
        if instance_mode:
            self.contour_tensors = [get_contour_maps(item, 'generated_contour_maps', contour_map_width) for item in
                                    self.file_list]
            self.lab_tensors = [Aug.binarisation(path_to_tensor(item[1], label=True)) for item in self.file_list]
        else:
            self.lab_tensors = [path_to_tensor(item[1], label=True) for item in self.file_list]
        self.img_tensors = [path_to_tensor(item[0], label=False) for item in self.file_list]
        if exclude_edge and not instance_mode:
            self.lab_tensors = [
                Aug.exclude_border_labels(item, exclude_edge_size_in, exclude_edge_size_out).to(torch.float32) for item
                in self.lab_tensors]
        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.train_multiplier = train_multiplier
        self.hw_size = hw_size
        self.d_size = d_size
        self.foreground_threshold = 0 if allow_negative else 0.001

        super().__init__()

    def __len__(self):
        return self.num_files * self.train_multiplier

    def __getitem__(self, idx):
        # Get the image and label tensors at the specified index
        # C, D, H, W
        idx = math.floor(idx / self.train_multiplier)
        img_tensor, lab_tensor = self.img_tensors[idx][None, :], self.lab_tensors[idx][None, :]
        if self.instance_mode:
            contour_tensor = self.contour_tensors[idx][None, :]
            img_tensor, lab_tensor, contour_tensor = apply_aug(img_tensor, lab_tensor, contour_tensor,
                                                               self.augmentation_params, self.hw_size, self.d_size,
                                                               self.foreground_threshold)
            return img_tensor, lab_tensor, contour_tensor
        else:
            img_tensor, lab_tensor = apply_aug(img_tensor, lab_tensor, None,
                                               self.augmentation_params, self.hw_size, self.d_size,
                                               self.foreground_threshold)
            return img_tensor, lab_tensor


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
    """

    def __init__(self, images_dir, hw_size, d_size, instance_mode, augmentation_csv, contour_map_width=1):
        # Get a list of file paths for images and labels
        file_list = make_dataset_tv(images_dir)
        self.num_files = len(file_list)
        self.instance_mode = instance_mode
        if instance_mode:
            tensors_pairs = [(path_to_tensor(item[0], label=False),
                              Aug.binarisation(path_to_tensor(item[1], label=True)),
                              get_contour_maps(item, 'generated_contour_maps', contour_map_width)) for item in file_list]
        else:
            tensors_pairs = [(path_to_tensor(item[0], label=False), path_to_tensor(item[1], label=True)) for item in
                             file_list]
        self.chopped_tensor_pairs = []
        # augmentation_params = pd.read_csv(augmentation_csv)
        # Crop the tensors, so they are the standard shape specified in the augmentation csv.
        for pairs in tensors_pairs:
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
                        cropped_img = pairs[0][depth_start:depth_end, height_start:height_end,
                                      width_start:width_end]
                        cropped_lab = pairs[1][depth_start:depth_end, height_start:height_end,
                                      width_start:width_end]
                        if instance_mode:
                            cropped_contour = pairs[2][depth_start:depth_end,
                                              height_start:height_end,
                                              width_start:width_end]
                            self.chopped_tensor_pairs.append((cropped_img, cropped_lab, cropped_contour))
                        else:
                            self.chopped_tensor_pairs.append((cropped_img, cropped_lab))
        super().__init__()

    def __len__(self):
        return self.num_files * self.total_multiplier

    def __getitem__(self, idx):
        img_tensor, lab_tensor = self.chopped_tensor_pairs[idx][0][None, :], self.chopped_tensor_pairs[idx][1][None, :]
        if self.instance_mode:
            contour_tensor = self.chopped_tensor_pairs[idx][2][None, :]
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
    """

    def __init__(self, images_dir, hw_size=128, depth_size=128, hw_overlap=16, depth_overlap=16, TTA_hw=False,
                 leave_out_idx=None):
        self.file_list = make_dataset_predict(images_dir)
        self.patches_list = []
        self.meta_list = []
        if leave_out_idx is not None:
            file = self.file_list[leave_out_idx]
            self.file_list = [file]
        for file in self.file_list:
            image = path_to_tensor(file, label=False)[None, :]
            image_list = [image]
            if TTA_hw:
                image_list.append(T_F.hflip(image))
                image_list.append(T_F.vflip(image))
                image_list.append(T_F.vflip(T_F.hflip(image)))
            file_name = os.path.basename(file)

            depth = image.shape[1]
            height = image.shape[2]
            width = image.shape[3]
            # Calculate the multipliers for padding and cropping
            depth_multiplier = math.ceil(depth / depth_size)
            height_multiplier = math.ceil(height / hw_size)
            width_multiplier = math.ceil(width / hw_size)
            TTA_multiplier = len(image_list)
            # Padding and cropping
            paddings = (hw_overlap, hw_overlap,
                        hw_overlap, hw_overlap,
                        depth_overlap, depth_overlap)
            # Experiments show replicate works best compare to constant or reflect
            padded_image_list = []
            for item in image_list:
                item = F.pad(item, paddings, mode="replicate")
                padded_image_list.append(item)
            del image_list, item
            # Loop through each depth, height, and width index
            for i in range(TTA_multiplier):
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
                            patch = padded_image_list[i][:,
                                                         depth_start:depth_end,
                                                         height_start:height_end,
                                                         width_start:width_end]
                            self.patches_list.append(patch)
            self.meta_list.append((file_name, image.shape))
        super().__init__()

    def __len__(self):
        return len(self.patches_list)

    def __getitem__(self, idx):
        return self.patches_list[idx]

    def __getmetainfo__(self):
        return self.meta_list


class CollectedDataset(torch.utils.data.Dataset):
    """
    For handling dataset pairing.
    """
    def __init__(self, dataset_positive, dataset_negative):
        self.dataset_positive = dataset_positive
        self.dataset_negative = dataset_negative

    def __len__(self):
        return len(self.dataset_positive) + len(self.dataset_negative)

    def __getitem__(self, idx):
        if idx < len(self.dataset_positive):
            pos = self.dataset_positive[idx]
            return pos
        else:
            idx = idx - len(self.dataset_positive)
            return self.dataset_negative[idx]


class CollectedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        super(CollectedSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.indices = np.arange(len(data_source))

    def __iter__(self):
        if self.batch_size == 1:
            return iter(np.random.permutation(len(self.data_source)))
        else:
            original_array = np.arange(len(self.data_source))
            pairs = [(original_array[i], original_array[i + 1]) for i in range(0, len(self.data_source), 2)]
            np.random.shuffle(pairs)
            shuffled_array = np.array([elem for pair in pairs for elem in pair])
            return iter(shuffled_array)

    def __len__(self):
        return len(self.data_source)


def custom_collate(batch):
    if len(batch) == 1:
        return batch[0]
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
        contour_img = Aug.instance_contour_transform(path_to_tensor(img_path, label=True), contour_outward=contour_map_width)
        array = np.asarray(contour_img)
        imageio.v3.imwrite(uri=f'{contour_img_path}', image=np.uint8(array))
        print(f'Saved {contour_img_name}')
    else:
        contour_img = path_to_tensor(contour_img_path, label=True)
        print(
            f'Loaded previously saved {contour_img_name}. Remember to delete old one if you made change to the label!')
    return contour_img


class CrossValidationDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, augmentation_csv, hw_size, d_size, leave_out_index=0, mode='Train', train_multiplier=1):
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        self.leave_out = self.file_list.pop(leave_out_index)
        self.img_tensors = [path_to_tensor(item[0], label=False) for item in self.file_list]
        self.lab_tensors = [path_to_tensor(item[1], label=True) for item in self.file_list]
        self.train_multiplier = train_multiplier

        self.leave_out_img = path_to_tensor(self.leave_out[0], label=False)[None, :]
        leave_out_img_name = os.path.basename(self.leave_out[0])
        self.leave_out_label = path_to_tensor(self.leave_out[1], label=True)[None, :]
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


def stitch_output_volumes(tensor_list, meta_list, hw_size=128, depth_size=128, hw_overlap=16, depth_overlap=16,
                          TTA_hw=False):
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
    seperator = 0
    TTA_multiplier = 1
    if TTA_hw:
        TTA_multiplier *= 4

    for meta_info in meta_list:
        depth = meta_info[1][1]
        height = meta_info[1][2]
        width = meta_info[1][3]
        file_name = "Mask_" + meta_info[0]
        depth_multiplier = math.ceil(depth / depth_size)
        height_multiplier = math.ceil(height / hw_size)
        width_multiplier = math.ceil(width / hw_size)
        total_multiplier = depth_multiplier * height_multiplier * width_multiplier * TTA_multiplier
        TTA_list = []
        for TTA_idx in range(TTA_multiplier):
            result_volume = torch.zeros((depth_multiplier * depth_size,
                                         height_multiplier * hw_size,
                                         width_multiplier * hw_size), dtype=torch.float32)
            for i in range(seperator + TTA_idx * (total_multiplier // TTA_multiplier),
                           seperator + (TTA_idx + 1) * (total_multiplier // TTA_multiplier)):
                if hw_overlap != 0 and depth_overlap != 0:
                    tensor_work_with = tensor_list[i][depth_overlap:-depth_overlap,
                                       hw_overlap:-hw_overlap,
                                       hw_overlap:-hw_overlap]
                elif hw_overlap != 0 and depth_overlap == 0:
                    tensor_work_with = tensor_list[i][:,
                                       hw_overlap:-hw_overlap,
                                       hw_overlap:-hw_overlap]
                elif hw_overlap == 0 and depth_overlap != 0:
                    tensor_work_with = tensor_list[i][depth_overlap:-depth_overlap,
                                       :,
                                       :]
                else:
                    tensor_work_with = tensor_list[i]

                depth_idx = math.floor(i / (height_multiplier * width_multiplier)) % depth_multiplier
                height_idx = math.floor(i / (width_multiplier)) % height_multiplier
                width_idx = (i) % width_multiplier

                if depth_multiplier > 1:
                    depth_start = (depth_size - (
                            (depth_size * depth_multiplier - depth) / (depth_multiplier - 1))) * depth_idx
                    depth_start = math.floor(depth_start)
                else:
                    depth_start = 0
                depth_end = depth_start + depth_size
                if height_multiplier > 1:
                    height_start = (hw_size - (
                            (hw_size * height_multiplier - height) / (height_multiplier - 1))) * height_idx
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
                result_volume[depth_start:depth_end, height_start:height_end, width_start:width_end] = tensor_work_with
            result_volume = result_volume[0:depth, 0:height, 0:width]
            if TTA_idx == 1:
                result_volume = T_F.hflip(result_volume)
            elif TTA_idx == 2:
                result_volume = T_F.vflip(result_volume)
            elif TTA_idx == 3:
                result_volume = T_F.vflip(T_F.hflip(result_volume))
            TTA_list.append(result_volume)
            # array = np.asarray(result_volume)
            # imageio.v3.imwrite(uri=f'debug_output/{TTA_idx}.tiff', image=np.float32(array))
        result_volume = torch.mean(torch.stack(TTA_list, dim=0), dim=0)
        seperator += total_multiplier
        # array = np.asarray(result_volume)
        # imageio.v3.imwrite(uri=f'TTA Output.tiff', image=np.float32(array))
        output_list.append((result_volume, file_name))
    return output_list


def predictions_to_final_img(predictions, meta_list, direc, hw_size=128, depth_size=128, hw_overlap=16,
                             depth_overlap=16, TTA_hw=False):
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
    tensor_list = []
    for prediction in predictions:  # 将这些输出张量项目从predictions里拿出来
        # 分割prediction，形成包含了多个元素（图片张量）的元组，每个元素的Batch维度都是1
        splitted = torch.split(prediction, split_size_or_sections=1, dim=0)
        for single_tensor in splitted:  # 将这个元组里每个单独元素（图片张量）拆分出来
            # It appears that if the final output is a volume, I will need to squeeze the first dimension(Batch)
            single_tensor = torch.squeeze(single_tensor, (0, 1))
            tensor_list.append(single_tensor)
    del splitted, single_tensor
    stitched_volumes = stitch_output_volumes(tensor_list, meta_list, hw_size, depth_size, hw_overlap, depth_overlap,
                                             TTA_hw)
    del tensor_list
    for volume in stitched_volumes:
        array = np.asarray(volume[0])
        array = np.where(array >= 0.5, 1, 0)
        imageio.v3.imwrite(uri=f'{direc}/{volume[1]}', image=np.uint8(array))


def predictions_to_final_img_instance(predictions, meta_list, direc, hw_size=128, depth_size=128, hw_overlap=16,
                                      depth_overlap=16, TTA_hw=False):
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
    """
    tensor_list_p = []
    tensor_list_c = []
    for prediction in predictions:
        splitted_p = torch.split(prediction[0], split_size_or_sections=1, dim=0)
        splitted_c = torch.split(prediction[1], split_size_or_sections=1, dim=0)
        for single_tensor in splitted_p:
            single_tensor = torch.squeeze(single_tensor, (0, 1))
            tensor_list_p.append(single_tensor)
        del splitted_p
        for single_tensor in splitted_c:
            single_tensor = torch.squeeze(single_tensor, (0, 1))
            tensor_list_c.append(single_tensor)
        del splitted_c
    del predictions
    stitched_volumes_p = stitch_output_volumes(tensor_list_p, meta_list, hw_size, depth_size, hw_overlap, depth_overlap,
                                               TTA_hw)
    del tensor_list_p
    stitched_volumes_c = stitch_output_volumes(tensor_list_c, meta_list, hw_size, depth_size, hw_overlap, depth_overlap,
                                               TTA_hw)
    del tensor_list_c
    #    for volume in stitched_volumes_p:
    #        array = np.asarray(volume[0])
    #        imageio.v3.imwrite(uri=f'{direc}/Pixels_{volume[1]}', image=np.uint8(array))
    #    for volume in stitched_volumes_c:
    #        array = np.asarray(volume[0])
    #        imageio.v3.imwrite(uri=f'{direc}/Contour_{volume[1]}', image=np.float32(array))
    for semantic, contour in zip(stitched_volumes_p, stitched_volumes_c):
        semantic_array = np.asarray(semantic[0])
        semantic_array = np.where(semantic_array >= 0.5, 1, 0)
        contour_array = np.asarray(contour[0])
        imageio.v3.imwrite(uri=f'{direc}/Pixels_{semantic[1]}', image=np.uint8(semantic_array))
        imageio.v3.imwrite(uri=f'{direc}/Contour_{contour[1]}', image=np.float32(contour_array))
        print(
            f'Computing instance segmentation using contour data for {contour[1]}... Can take a while if the image is big.')
        instance_result = instance_segmentation_simple(semantic[0], contour[0])
        instance_array = np.asarray(instance_result)
        imageio.v3.imwrite(uri=f'{direc}/Instance_{contour[1]}', image=np.uint16(instance_array))


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


def instance_segmentation_simple(semantic_map, contour_map, size_threshold=10, distance_threshold=2):
    """
    Using a semantic segmentation map and a contour map to separate touching objects and perform instance segmentation.
    Pixels in touching areas are assigned to the closest object based on the largest proportion of pixels within 5 pixel distance to the pixel.

    Args:
        semantic_map (np.Array): The input semantic segmented map.
        contour_map (np.Array): The input contour segmented map.
        size_threshold (int): The minimal size in pixel of each object. Object smaller than this will be removed.
        distance_threshold (int): The radius in pixels to search for nearby pixels when allocating. Default: 2.

    Returns:
        torch.Tensor: instance segmented map where 0 is background and every other value represent an object.
    """
    semantic_map = torch.where(semantic_map >= 0.5, True, False)
    # Treat contour map with lower threshold to ensure separation
    contour_map = torch.where(contour_map >= 0.35, True, False)

    # Find boundary that lies within foreground objects
    touching_map = torch.logical_and(contour_map, semantic_map)
    # Remove touching area between foreground objects
    new_map = torch.logical_xor(semantic_map, touching_map)
    del semantic_map, contour_map

    structure = torch.ByteTensor([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    ])
    # Connected Component Labelling
    segmentation, _ = label(new_map.numpy(), structure=structure.numpy())
    del structure, new_map
    # Remove small segments
    segmentation = morphology.remove_small_objects(segmentation, min_size=size_threshold, connectivity=2)
    segmentation = torch.tensor(segmentation, dtype=torch.int16)

    # Pre-compute all local segments
    touching_pixels = torch.nonzero(touching_map, as_tuple=False).to(torch.int32)
    del touching_map
    num_touching_pixels = len(touching_pixels)
    minlength = segmentation.max().item()
    map_size = segmentation.shape
    new_segmentation = segmentation.clone()

    # Allocates the lost pixels to the object that has the largest proportion of pixels within the threshold distance
    def allocate_pixels(pixel_idx):
        z, y, x = touching_pixels[pixel_idx, 0], touching_pixels[pixel_idx, 1], touching_pixels[pixel_idx, 2]
        local_segment = segmentation[
                        max(z - distance_threshold, 0):min(z + distance_threshold, map_size[0] - 1),
                        max(y - distance_threshold, 0):min(y + distance_threshold, map_size[1] - 1),
                        max(x - distance_threshold, 0):min(x + distance_threshold, map_size[2] - 1)
                        ]
        # Compute object counts with background count included
        object_counts = torch.bincount(local_segment.flatten(), minlength=minlength)
        # Exclude background count
        object_counts = object_counts[1:]
        if object_counts.numel() > 0:
            closest_objects = torch.argmax(object_counts, dim=0).to(torch.int16) + 1
            new_segmentation[touching_pixels[pixel_idx, 0],
                             touching_pixels[pixel_idx, 1],
                             touching_pixels[pixel_idx, 2]] = closest_objects

    data_loader = torch.utils.data.DataLoader(range(num_touching_pixels), batch_size=1, num_workers=0, pin_memory=True)
    start_time = time.time()
    current_proportion = 0.0
    for batch in data_loader:
        allocate_pixels(batch)
        if (batch[0] + 1) / len(touching_pixels) >= current_proportion + 0.001:
            elapsed_time = time.time() - start_time
            print(
                f"Processed {batch[0] + 1}/{len(touching_pixels)} voxels at {((batch[0] + 1) / elapsed_time):.2f} voxels/sec")
            current_proportion += 0.001

    return new_segmentation


# 这里的函数用于测试各个组件是否能正常运作
if __name__ == "__main__":
    # fake_predictions = [torch.randn(4, 512, 512), torch.randn(4, 512, 512)]
    # predictions_to_final_img(fake_predictions)
    test_tensor = path_to_tensor("../test_img.tif")
    print(test_tensor)
    # print(test_loader.dataset[0][0])
    # test_tensor = path_to_tensor('Datasets/train/lab/Labels_Jul13ab_nntrain_1.tif', label=True)
    # print(test_tensor.shape)
    # test_tensor = composed_transform(test_tensor, 1)
    # print(test_tensor.shape)
    # print(test_tensor)
    # test_array = np.asarray(test_tensor)
    # im = imageio.volsave('Result.tif', test_array)
