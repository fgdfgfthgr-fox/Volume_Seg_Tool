import math
import os

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms.v2.functional as T_F
import numpy as np
import imageio
import random
import scipy
import pandas as pd
from joblib import Parallel, delayed
from skimage import morphology
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label
from skimage.feature import peak_local_max
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
    if not label:
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
            img = (img/max_value).astype(np.float32)
    return torch.from_numpy(img)


def binarisation(tensor):
    """
    A quick and dirty way to convert an instance labelled tensor to a semantic labelled tensor.
    """
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def apply_aug(img_tensor, lab_tensor, augmentation_params):
    """
    Apply Image Augmentations to an image tensor and its label tensor using the augmentation parameters from a DataFrame.

    Args:
        img_tensor (torch.Tensor): Image tensor. Should be float32.
        lab_tensor (torch.Tensor): Label tensor, should be the same shape as img_tensor.
        augmentation_params (DataFrame): The DataFrame which the augmentation parameters will be used from.

    Returns:
        Transformed Image and Label Tensor.
    """
    for _, row in augmentation_params.iterrows():
        k = row['Augmentation']
        if k == 'Image Depth':
            depth = int(row['Value'])
        elif k == 'Image Height':
            height = int(row['Value'])
        elif k == 'Image Width':
            width = int(row['Value'])
        augmentation_method, prob = row['Augmentation'], row['Probability']
        if augmentation_method == 'Rescaling':
            if random.random() < prob:
                scale = random.uniform(row['Low Bound'], row['High Bound'])
                img_tensor, lab_tensor = Aug.custom_rand_crop([img_tensor, lab_tensor],
                                                              int(scale * depth),
                                                              int(scale * height),
                                                              int(scale * width))
                img_tensor = img_tensor[None, :]
                img_tensor = F.interpolate(img_tensor, size=(depth, height, width), mode="trilinear",
                                           align_corners=True)
                lab_tensor = lab_tensor[None, :]
                lab_tensor = F.interpolate(lab_tensor, size=(depth, height, width), mode="nearest-exact")
                img_tensor = torch.squeeze(img_tensor, 0)
                lab_tensor = torch.squeeze(lab_tensor, 0)
            else:
                img_tensor, lab_tensor = Aug.custom_rand_crop([img_tensor, lab_tensor],
                                                              depth,
                                                              height,
                                                              width)
        elif augmentation_method == 'Rotate xy' and random.random() < prob:
            img_tensor, lab_tensor = Aug.random_rotation_3d((img_tensor, lab_tensor), interpolations=('bilinear', 'nearest'),
                                                            plane='xy', angle_range=(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate xz' and random.random() < prob:
            img_tensor, lab_tensor = Aug.random_rotation_3d((img_tensor, lab_tensor), interpolations=('bilinear', 'nearest'),
                                                            plane='xz', angle_range=(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate yz' and random.random() < prob:
            img_tensor, lab_tensor = Aug.random_rotation_3d((img_tensor, lab_tensor), interpolations=('bilinear', 'nearest'),
                                                            plane='yz', angle_range=(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Vertical Flip' and random.random() < prob:
            img_tensor, lab_tensor = T_F.vflip(img_tensor), T_F.vflip(lab_tensor)
        elif augmentation_method == 'Horizontal Flip' and random.random() < prob:
            img_tensor, lab_tensor = T_F.hflip(img_tensor), T_F.hflip(lab_tensor)
        elif augmentation_method == 'Simulate Low Resolution' and random.random() < prob:
            img_tensor = Aug.sim_low_res(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Blur' and random.random() < prob:
            img_tensor = Aug.gaussian_blur_3d(img_tensor, int(row['Value']))
        elif augmentation_method == 'Gradient Gamma' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']))
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
            lab_tensor = Aug.gaussian_blur_3d(lab_tensor, int(row['Value']))
    return img_tensor, lab_tensor


def apply_aug_instance(img_tensor, lab_tensor, contour_tensor, augmentation_params):
    """
    Apply Image Augmentations to an image tensor, its label tensor and its contour tensor
    using the augmentation parameters from a DataFrame.

    Args:
        img_tensor (torch.Tensor): Image tensor. Should be float32.
        lab_tensor (torch.Tensor): Label tensor, should be the same shape as img_tensor.
        contour_tensor (torch.Tensor): Contour tensor, should be the same shape as img_tensor.
        augmentation_params (DataFrame): The DataFrame which the augmentation parameters will be used from.

    Returns:
        Transformed Image and Label Tensor.
    """
    for _, row in augmentation_params.iterrows():
        k = row['Augmentation']
        if k == 'Image Depth':
            depth = int(row['Value'])
        elif k == 'Image Height':
            height = int(row['Value'])
        elif k == 'Image Width':
            width = int(row['Value'])
        augmentation_method, prob = row['Augmentation'], row['Probability']
        if augmentation_method == 'Rescaling':
            if random.random() < prob:
                scale = random.uniform(row['Low Bound'], row['High Bound'])
                img_tensor, lab_tensor, contour_tensor = Aug.custom_rand_crop([img_tensor, lab_tensor, contour_tensor],
                                                                              int(scale * depth),
                                                                              int(scale * height),
                                                                              int(scale * width))
                img_tensor = img_tensor[None, :]
                img_tensor = F.interpolate(img_tensor, size=(depth, height, width), mode="trilinear",
                                           align_corners=True)
                lab_tensor = lab_tensor[None, :]
                lab_tensor = F.interpolate(lab_tensor, size=(depth, height, width), mode="nearest-exact")
                contour_tensor = contour_tensor[None, :]
                contour_tensor = F.interpolate(contour_tensor, size=(depth, height, width), mode="nearest-exact")
                img_tensor = torch.squeeze(img_tensor, 0)
                lab_tensor = torch.squeeze(lab_tensor, 0)
                contour_tensor = torch.squeeze(contour_tensor, 0)
            else:
                img_tensor, lab_tensor, contour_tensor = Aug.custom_rand_crop([img_tensor, lab_tensor, contour_tensor],
                                                                              depth,
                                                                              height,
                                                                              width)
        elif augmentation_method == 'Rotate xy' and random.random() < prob:
            img_tensor, lab_tensor, contour_tensor = Aug.random_rotation_3d((img_tensor, lab_tensor, contour_tensor),
                                                                            interpolations=('bilinear', 'nearest', 'nearest'), fill_values=(0,0,0),
                                                                            plane='xy', angle_range=(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate xz' and random.random() < prob:
            img_tensor, lab_tensor, contour_tensor = Aug.random_rotation_3d((img_tensor, lab_tensor, contour_tensor),
                                                                            interpolations=('bilinear', 'nearest', 'nearest'), fill_values=(0,0,0),
                                                                            plane='xz', angle_range=(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Rotate yz' and random.random() < prob:
            img_tensor, lab_tensor, contour_tensor = Aug.random_rotation_3d((img_tensor, lab_tensor, contour_tensor),
                                                                            interpolations=('bilinear', 'nearest', 'nearest'), fill_values=(0,0,0),
                                                                            plane='yz', angle_range=(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Vertical Flip' and random.random() < prob:
            img_tensor, lab_tensor, contour_tensor = T_F.vflip(img_tensor), T_F.vflip(lab_tensor), T_F.vflip(contour_tensor)
        elif augmentation_method == 'Horizontal Flip' and random.random() < prob:
            img_tensor, lab_tensor, contour_tensor = T_F.hflip(img_tensor), T_F.hflip(lab_tensor), T_F.hflip(contour_tensor)
        elif augmentation_method == 'Simulate Low Resolution' and random.random() < prob:
            img_tensor = Aug.sim_low_res(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Gaussian Blur' and random.random() < prob:
            img_tensor = Aug.gaussian_blur_3d(img_tensor, int(row['Value']))
        elif augmentation_method == 'Gradient Gamma' and random.random() < prob:
            img_tensor = Aug.random_gradient(img_tensor, (row['Low Bound'], row['High Bound']))
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
            lab_tensor = Aug.gaussian_blur_3d(lab_tensor, int(row['Value']))
        elif augmentation_method == 'Contour Blur' and random.random() < prob:
            contour_tensor = contour_tensor.to(torch.float32)
            contour_tensor = Aug.gaussian_blur_3d(contour_tensor, int(row['Value']))
    return img_tensor, lab_tensor, contour_tensor


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
    A torch.utils.data.Dataset class that handles the training dataset of semantic segmentation.

    Args:
        images_dir (str): Path to the directory where images are stored.
        augmentation_csv (str): Path to the .csv file that contains the image augmentations parameters.
        train_multiplier (int): A.k.a repeat, the number of training images in each epoch are multiplied by this number. Default: 1.
        exclude_edge (bool): If true, the borders of the objects in the label tensor will be excluded from gradient calculation.
                             Default: false.
        exclude_edge_size_in (int): The thickness of the border in pixels, toward the inside of each object. Default: 1.
        exclude_edge_size_out (int): The thickness of the border in pixels, toward the outside of each object. Default: 1.
    """
    def __init__(self, images_dir, augmentation_csv, train_multiplier=1,
                 exclude_edge=False, exclude_edge_size_in=1, exclude_edge_size_out=1):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        self.img_tensors = [path_to_tensor(item[0], label=False) for item in self.file_list]
        self.lab_tensors = [path_to_tensor(item[1], label=True) for item in self.file_list]
        if exclude_edge:
            self.lab_tensors = [Aug.exclude_border_labels(item, exclude_edge_size_in, exclude_edge_size_out).to(torch.float32) for item in self.lab_tensors]
        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.train_multiplier = train_multiplier

        super().__init__()

    def __len__(self):
        return self.num_files * self.train_multiplier

    def __getitem__(self, idx):
        # Get the image and label tensors at the specified index
        # C, D, H, W
        idx = math.floor(idx / self.train_multiplier)
        img_tensor, lab_tensor = self.img_tensors[idx][None, :], self.lab_tensors[idx][None, :]
        img_tensor, lab_tensor = apply_aug(img_tensor, lab_tensor, self.augmentation_params)
        lab_tensor = lab_tensor.squeeze(1)
        return img_tensor, lab_tensor


class ValDataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the validation dataset of semantic segmentation.
    Note if the image is larger than the size specified in the augmentation_csv, it will get cropped into several
    (potentially) overlapping smaller images. The validation result will be the average of these smaller images.

    Args:
        images_dir (str): Path to the directory where images are stored.
        augmentation_csv (str): Path to the .csv file that contains the image augmentations parameters.
    """
    def __init__(self, images_dir, augmentation_csv):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        tensors_pairs = [(path_to_tensor(item[0], label=False), path_to_tensor(item[1], label=True)) for item in self.file_list]
        self.chopped_tensor_pairs = []
        self.augmentation_params = pd.read_csv(augmentation_csv)
        for _, row in self.augmentation_params.iterrows():
            k = row['Augmentation']
            if k == 'Image Depth':
                depth = int(row['Value'])
            elif k == 'Image Height':
                height = int(row['Value'])
            elif k == 'Image Width':
                width = int(row['Value'])
        # Crop the tensors, so they are the standard shape specified in the augmentation csv.
        for pairs in tensors_pairs:
            self.depth, self.height, self.width = pairs[0].shape
            depth_multiplier = math.ceil(self.depth / depth)
            height_multiplier = math.ceil(self.height / height)
            width_multiplier = math.ceil(self.width / width)
            self.total_multiplier = depth_multiplier * height_multiplier * width_multiplier
            self.leave_out_list = []
            # Loop through each depth, height, and width index
            for depth_idx in range(depth_multiplier):
                for height_idx in range(height_multiplier):
                    for width_idx in range(width_multiplier):
                        # Calculate the start and end indices for depth, height, and width
                        if depth_multiplier > 1:
                            depth_start = (depth - ((depth * depth_multiplier - self.depth) / (depth_multiplier - 1))) * depth_idx
                            depth_start = math.floor(depth_start)
                        else:
                            depth_start = 0
                        depth_end = depth_start + depth
                        if height_multiplier > 1:
                            height_start = (height - ((height * height_multiplier - self.height) / (height_multiplier - 1))) * height_idx
                            height_start = math.floor(height_start)
                        else:
                            height_start = 0
                        height_end = height_start + height
                        if width_multiplier > 1:
                            width_start = (width - ((width * width_multiplier - self.width) / (width_multiplier - 1))) * width_idx
                            width_start = math.floor(width_start)
                        else:
                            width_start = 0
                        width_end = width_start + width
                        cropped_img = pairs[0][depth_start:depth_end, height_start:height_end,
                                               width_start:width_end]
                        cropped_lab = pairs[1][depth_start:depth_end, height_start:height_end,
                                               width_start:width_end]
                        self.chopped_tensor_pairs.append((cropped_img, cropped_lab))
        super().__init__()

    def __len__(self):
        return self.num_files * self.total_multiplier

    def __getitem__(self, idx):
        # Get the image and label tensors at the specified index
        # C, D, H, W
        img_tensor, lab_tensor = self.chopped_tensor_pairs[idx][0][None, :], self.chopped_tensor_pairs[idx][1][None, :]
        return img_tensor, lab_tensor


# 自定义的数据集结构，用于存储预测数据
class Predict_Dataset(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the predict dataset of semantic segmentation.\n
    Note if the image is larger than the size specified in the augmentation_csv, it will get cropped into several
    (potentially) overlapping smaller images. The final results will be stitched together from predictions of these smaller images.\n
    The actual height and width of each patch is hw_overlap + 2 * depth_overlap. Same goes for depth.

    Args:
        images_dir (str): Path to the directory where images are stored.
        hw_size (int): The height and width patches the prediction image will be cropped to. In pixels.
        depth_size (int): The depth of patches the prediction image will be cropped to. In pixels.
        hw_overlap (int): The additional gain in height and width of the patches. In pixels. Helps smooth out borders between patches.
        depth_overlap (int): The additional gain in depth of the patches. In pixels. Helps smooth out borders between patches.
    """
    def __init__(self, images_dir, hw_size=128, depth_size=128, hw_overlap=16, depth_overlap=16):
        self.file_list = make_dataset_predict(images_dir)
        self.hw_size = hw_size
        self.depth_size = depth_size
        self.hw_overlap = hw_overlap
        self.depth_overlap = depth_overlap
        self.patches_list = []
        self.meta_list = []
        for file in self.file_list:
            image = path_to_tensor(file, label=False)
            file_name = os.path.basename(file)
            depth = image.shape[0]
            height = image.shape[1]
            width = image.shape[2]
            # Calculate the multipliers for padding and cropping
            depth_multiplier = math.ceil(depth / self.depth_size)
            height_multiplier = math.ceil(height / self.hw_size)
            width_multiplier = math.ceil(width / self.hw_size)
            #total_multiplier = depth_multiplier * height_multiplier * width_multiplier
            # Padding and cropping
            paddings = (hw_overlap, hw_overlap,
                        hw_overlap, hw_overlap,
                        depth_overlap, depth_overlap)
            image = image[None, :]
            # Experiments show replicate works best compare to constant or reflect
            padded_image = F.pad(image, paddings, mode="replicate")
            # Loop through each depth, height, and width index
            for depth_idx in range(depth_multiplier):
                for height_idx in range(height_multiplier):
                    for width_idx in range(width_multiplier):
                        if depth_multiplier > 1:
                            depth_start = (depth_size - ((depth_size * depth_multiplier - depth) / (depth_multiplier - 1))) * depth_idx
                            depth_start = math.floor(depth_start)
                        else:
                            depth_start = 0
                        depth_end = depth_start + depth_size + (2 * depth_overlap)
                        if height_multiplier > 1:
                            height_start = (hw_size - ((hw_size * height_multiplier - height) / (height_multiplier - 1))) * height_idx
                            height_start = math.floor(height_start)
                        else:
                            height_start = 0
                        height_end = height_start + hw_size + (2 * hw_overlap)
                        if width_multiplier > 1:
                            width_start = (hw_size - ((hw_size * width_multiplier - width) / (width_multiplier - 1))) * width_idx
                            width_start = math.floor(width_start)
                        else:
                            width_start = 0
                        width_end = width_start + hw_size + (2 * hw_overlap)
                        patch = padded_image[:, depth_start:depth_end, height_start:height_end, width_start:width_end].to(torch.float32)
                        self.patches_list.append(patch)
            self.meta_list.append((file_name, image.shape))
        super().__init__()

    def __len__(self):
        return len(self.patches_list)

    def __getitem__(self, idx):
        return self.patches_list[idx]

    def __getmetainfo__(self):
        return self.meta_list


class TrainDatasetInstance(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the training dataset of instance segmentation.

    Args:
        images_dir (str): Path to the directory where images are stored.
        augmentation_csv (str): Path to the .csv file that contains the image augmentations parameters.
        train_multiplier (int): A.k.a repeat, the number of training images in each epoch are multiplied by this number. Default: 1.
    """
    def __init__(self, images_dir, augmentation_csv, train_multiplier=1):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        self.img_tensors = [path_to_tensor(item[0], label=False) for item in self.file_list]
        self.lab_tensors = [binarisation(path_to_tensor(item[1], label=True)) for item in self.file_list]
        print('Generating contour map(s)...')
        self.contour_tensors = [instance_contour_transform(path_to_tensor(item[1], label=True)) for item in self.file_list]
        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.train_multiplier = train_multiplier

        super().__init__()

    def __len__(self):
        return self.num_files * self.train_multiplier

    def __getitem__(self, idx):
        # Get the image and label tensors at the specified index
        # C, D, H, W
        idx = math.floor(idx / self.train_multiplier)
        img_tensor, lab_tensor, contour_tensor = self.img_tensors[idx][None, :], self.lab_tensors[idx][None, :], self.contour_tensors[idx][None, :]
        img_tensor, lab_tensor, contour_tensor = apply_aug_instance(img_tensor, lab_tensor, contour_tensor, self.augmentation_params)
        lab_tensor = lab_tensor.squeeze(1)
        return img_tensor, lab_tensor, contour_tensor


class ValDatasetInstance(torch.utils.data.Dataset):
    """
    A torch.utils.data.Dataset class that handles the validation dataset of instance segmentation.
    Note if the image is larger than the size specified in the augmentation_csv, it will get cropped into several
    (potentially) overlapping smaller images. The validation result will be the average of these smaller images.

    Args:
        images_dir (str): Path to the directory where images are stored.
        augmentation_csv (str): Path to the .csv file that contains the image augmentations parameters.
    """
    def __init__(self, images_dir, augmentation_csv):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        tensors_pairs = [(path_to_tensor(item[0], label=False),
                          binarisation(path_to_tensor(item[1], label=True)),
                          instance_contour_transform(path_to_tensor(item[1], label=True))) for item in self.file_list]
        self.chopped_tensor_pairs = []
        self.augmentation_params = pd.read_csv(augmentation_csv)
        for _, row in self.augmentation_params.iterrows():
            k = row['Augmentation']
            if k == 'Image Depth':
                depth = int(row['Value'])
            elif k == 'Image Height':
                height = int(row['Value'])
            elif k == 'Image Width':
                width = int(row['Value'])
        for pairs in tensors_pairs:
            self.depth, self.height, self.width = pairs[0].shape
            depth_multiplier = math.ceil(self.depth / depth)
            height_multiplier = math.ceil(self.height / height)
            width_multiplier = math.ceil(self.width / width)
            self.total_multiplier = depth_multiplier * height_multiplier * width_multiplier
            self.leave_out_list = []
            # Loop through each depth, height, and width index
            for depth_idx in range(depth_multiplier):
                for height_idx in range(height_multiplier):
                    for width_idx in range(width_multiplier):
                        # Calculate the start and end indices for depth, height, and width
                        if depth_multiplier > 1:
                            depth_start = (depth - ((depth * depth_multiplier - self.depth) / (depth_multiplier - 1))) * depth_idx
                            depth_start = math.floor(depth_start)
                        else:
                            depth_start = 0
                        depth_end = depth_start + depth
                        if height_multiplier > 1:
                            height_start = (height - ((height * height_multiplier - self.height) / (height_multiplier - 1))) * height_idx
                            height_start = math.floor(height_start)
                        else:
                            height_start = 0
                        height_end = height_start + height
                        if width_multiplier > 1:
                            width_start = (width - ((width * width_multiplier - self.width) / (width_multiplier - 1))) * width_idx
                            width_start = math.floor(width_start)
                        else:
                            width_start = 0
                        width_end = width_start + width
                        cropped_img = pairs[0][depth_start:depth_end, height_start:height_end,
                                               width_start:width_end]
                        cropped_lab = pairs[1][depth_start:depth_end, height_start:height_end,
                                               width_start:width_end]
                        cropped_contour = pairs[2][depth_start:depth_end, height_start:height_end,
                                                   width_start:width_end]
                        self.chopped_tensor_pairs.append((cropped_img, cropped_lab, cropped_contour))
        super().__init__()

    def __len__(self):
        return self.num_files * self.total_multiplier

    def __getitem__(self, idx):
        # Get the image and label tensors at the specified index
        # C, D, H, W
        img_tensor, lab_tensor, contour_tensor = self.chopped_tensor_pairs[idx][0][None, :], self.chopped_tensor_pairs[idx][1][None, :], self.chopped_tensor_pairs[idx][2][None, :]
        return img_tensor, lab_tensor, contour_tensor


'''
class Cross_Validation_Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, augmentation_csv, leave_out_index=0, train_mode=True, train_multiplier=1):
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        self.leave_out = self.file_list.pop(leave_out_index)
        self.img_tensors = [path_to_tensor(item[0], label=False) for item in self.file_list]
        self.lab_tensors = [path_to_tensor(item[1], label=True) for item in self.file_list]
        self.train_multiplier = train_multiplier

        self.leave_out_img = path_to_tensor(self.leave_out[0], label=False)
        self.leave_out_label = path_to_tensor(self.leave_out[1], label=True)
        self.train_mode = train_mode
        self.augmentation_params = pd.read_csv(augmentation_csv)
        for _, row in self.augmentation_params.iterrows():
            k = row['Augmentation']
            if k == 'Image Depth':
                depth = int(row['Value'])
            elif k == 'Image Height':
                height = int(row['Value'])
            elif k == 'Image Width':
                width = int(row['Value'])

        self.depth, self.height, self.width = self.leave_out_img.shape
        depth_multiplier = math.ceil(self.depth / depth)
        height_multiplier = math.ceil(self.height / height)
        width_multiplier = math.ceil(self.width / width)
        self.total_multiplier = depth_multiplier * height_multiplier * width_multiplier
        self.leave_out_list = []
        # Loop through each depth, height, and width index
        for depth_idx in range(depth_multiplier):
            for height_idx in range(height_multiplier):
                for width_idx in range(width_multiplier):
                    # Calculate the start and end indices for depth, height, and width
                    if depth_multiplier > 1:
                        depth_start = math.floor((depth-((depth*depth_multiplier-self.depth)/(depth_multiplier-1)))*depth_idx)
                    else:
                        depth_start = 0
                    depth_end = depth_start + depth
                    if height_multiplier > 1:
                        height_start = math.floor((height-((height*height_multiplier-self.height)/(height_multiplier-1)))*height_idx)
                    else:
                        height_start = 0
                    height_end = height_start + height
                    if width_multiplier > 1:
                        width_start = math.floor((width-((width*width_multiplier-self.width)/(width_multiplier-1)))*width_idx)
                    else:
                        width_start = 0
                    width_end = width_start + width
                    cropped_img = self.leave_out_img[depth_start:depth_end, height_start:height_end,
                                                     width_start:width_end]
                    cropped_lab = self.leave_out_label[depth_start:depth_end, height_start:height_end,
                                                       width_start:width_end]
                    self.leave_out_list.append((cropped_img, cropped_lab))
        super().__init__()

    def __len__(self):
        if self.train_mode:
            return (self.num_files - 1) * self.train_multiplier
        else:
            return self.total_multiplier

    def __getitem__(self, idx):
        if self.train_mode == True:
            idx = math.floor(idx / self.train_multiplier)
            img_tensor, lab_tensor = self.img_tensors[idx][None, :], self.lab_tensors[idx][None, :]
            img_tensor, lab_tensor = apply_aug(img_tensor, lab_tensor, self.augmentation_params)
            lab_tensor = lab_tensor.squeeze(1)
        else:
            img_tensor, lab_tensor = self.leave_out_list[idx][0][None, :], self.leave_out_list[idx][1][None, :]
        return img_tensor, lab_tensor
'''


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
    seperator = 0
    for meta_info in meta_list:
        depth = meta_info[1][1]
        height = meta_info[1][2]
        width = meta_info[1][3]
        file_name = "Mask_" + meta_info[0]
        depth_multiplier = math.ceil(depth / depth_size)
        height_multiplier = math.ceil(height / hw_size)
        width_multiplier = math.ceil(width / hw_size)
        total_multiplier = depth_multiplier * height_multiplier * width_multiplier
        result_volume = torch.zeros((depth_multiplier * depth_size,
                                     height_multiplier * hw_size,
                                     width_multiplier * hw_size), dtype=torch.float32)
        for i in range(seperator, seperator+total_multiplier):
            tensors_in_1_layer = height_multiplier * width_multiplier
            depth_idx = math.floor(i / tensors_in_1_layer) % depth_multiplier
            height_idx = math.floor(i / width_multiplier) % height_multiplier
            width_idx = i % width_multiplier
            tensor_work_with = tensor_list[i][depth_overlap:-depth_overlap,
                                              hw_overlap:-hw_overlap,
                                              hw_overlap:-hw_overlap]
            if depth_multiplier > 1:
                depth_start = (depth_size - ((depth_size * depth_multiplier - depth) / (depth_multiplier - 1))) * depth_idx
                depth_start = math.floor(depth_start)
            else:
                depth_start = 0
            depth_end = depth_start + depth_size
            if height_multiplier > 1:
                height_start = (hw_size - ((hw_size * height_multiplier - height) / (height_multiplier - 1))) * height_idx
                height_start = math.floor(height_start)
            else:
                height_start = 0
            height_end = height_start + hw_size
            if width_multiplier > 1:
                width_start = (hw_size - ((hw_size * width_multiplier - width) / (width_multiplier - 1))) * width_idx
                width_start = math.floor(width_start)
            else:
                width_start = 0
            width_end = width_start + hw_size
            result_volume[depth_start:depth_end, height_start:height_end, width_start:width_end] = tensor_work_with
        seperator += total_multiplier
        result_volume = result_volume[0:depth, 0:height, 0:width]
        output_list.append((result_volume, file_name))
    return output_list


def predictions_to_final_img(predictions, meta_list, direc, hw_size=128, depth_size=128, hw_overlap=16, depth_overlap=16):
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
            list.append(tensor_list, single_tensor)

    stitched_volumes = stitch_output_volumes(tensor_list, meta_list, hw_size, depth_size, hw_overlap, depth_overlap)
    for volume in stitched_volumes:
        array = np.asarray(volume[0])
        imageio.v3.imwrite(uri=f'{direc}/{volume[1]}', image=np.uint8(array))


def predictions_to_final_img_instance(predictions, meta_list, direc, hw_size=128, depth_size=128, hw_overlap=16, depth_overlap=16):
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
            list.append(tensor_list_p, single_tensor)
        for single_tensor in splitted_c:
            single_tensor = torch.squeeze(single_tensor, (0, 1))
            list.append(tensor_list_c, single_tensor)

    stitched_volumes_p = stitch_output_volumes(tensor_list_p, meta_list, hw_size, depth_size, hw_overlap, depth_overlap)
    stitched_volumes_c = stitch_output_volumes(tensor_list_c, meta_list, hw_size, depth_size, hw_overlap, depth_overlap)
#    for volume in stitched_volumes_p:
#        array = np.asarray(volume[0])
#        imageio.v3.imwrite(uri=f'{direc}/Pixels_{volume[1]}', image=np.uint8(array))
#    for volume in stitched_volumes_c:
#        array = np.asarray(volume[0])
#        imageio.v3.imwrite(uri=f'{direc}/Contour_{volume[1]}', image=np.float32(array))
    for semantic, contour in zip(stitched_volumes_p, stitched_volumes_c):
        semantic_array = np.asarray(semantic[0])
        contour_array = np.asarray(contour[0])
        imageio.v3.imwrite(uri=f'{direc}/Pixels_{semantic[1]}', image=np.uint8(semantic_array))
        imageio.v3.imwrite(uri=f'{direc}/Contour_{contour[1]}', image=np.float32(contour_array))
        instance_result = instance_segmentation_simple(semantic[0], contour[0])
        instance_array = np.asarray(instance_result)
        imageio.v3.imwrite(uri=f'{direc}/Instance_{contour[1]}', image=np.uint8(instance_array))
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


def instance_contour_transform(input_tensor, contour_inward=0, contour_outward=1):
    """
    Transform an instance segmented map into a contour map.\n
    The contour is generated using morphological erosion and dilation.

    Args:
        input_tensor (torch.Tensor): The input instance segmented map with a shape of (D, H, W).
                                     0 is background and every other value is a distinct object.
        contour_inward (int): Size of morphological erosion.
        contour_outward (int): Size of morphological dilation.

    Returns:
        torch.Tensor: Contour map where 0 are background or inside of objects while 1 are the boundaries.
    """
    input_array = input_tensor.numpy()
    unique_values = np.unique(input_array)
    structuring_element = np.ones((3, 3, 3), dtype=np.byte)
    transformed_tensor = torch.zeros_like(input_tensor, dtype=torch.uint8)

    def process_object(value):
        if value == 0:
            return  # Skip background

        # Create a binary mask for the current object
        object_mask = (input_array == value)

        # Erosion
        if contour_inward >= 1:
            eroded_mask = scipy.ndimage.binary_erosion(object_mask, structuring_element, contour_inward).astype(np.byte)
        else:
            eroded_mask = object_mask

        # Dilation
        if contour_outward >= 1:
            dilated_mask = scipy.ndimage.binary_dilation(object_mask, structuring_element, contour_outward).astype(np.byte)
        else:
            dilated_mask = object_mask

        eroded_tensor, dilated_tensor = torch.from_numpy(eroded_mask), torch.from_numpy(dilated_mask)

        # Mark boundary area as one
        excluded_regions = (eroded_tensor == 0) & (dilated_tensor == 1)
        transformed_tensor[excluded_regions] = 1

    # Use joblib to parallelize the loop
    Parallel(backend='threading', n_jobs=-1)(delayed(process_object)(value) for value in unique_values)

    return transformed_tensor


def instance_segmentation_batch(semantic_maps_uint8, contour_maps_float32):
    result_list = []

    for semantic_map_uint8, contour_map_float32 in zip(semantic_maps_uint8, contour_maps_float32):
        result_list.append(instance_segmentation_simple(semantic_map_uint8, contour_map_float32))

    return result_list


def instance_segmentation(semantic_map_uint8, contour_map_float32, size_threshold=10):
    """
    WIP.
    """
    # Convert PyTorch tensors to NumPy arrays
    semantic_map_np = semantic_map_uint8.cpu().numpy()
    contour_map_np = contour_map_float32.cpu().numpy()

    # Distance transform the semantic map
    semantic_map_distance = distance_transform_edt(semantic_map_np)
    coords = peak_local_max(semantic_map_distance, labels=semantic_map_np, min_distance=10)
    mask = np.zeros(semantic_map_distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = label(mask)

    # Minus the boundary map from distance transform map to create a map that's high in the central of the object,
    # zero as background, and large negative at the boundary.
    new_map = semantic_map_distance - (contour_map_np * 10)
    # Reverse the map, so it's low in the central of the object, slowly elevate as it get close to boundary,
    # then sudden spike up at the boundary pixel. Background is still zero.
    new_map = -new_map

    # Apply watershed algorithm
    segmentation = watershed(new_map, markers, mask=semantic_map_np)

    # Remove small segments
    segmentation = morphology.remove_small_objects(segmentation, min_size=size_threshold, connectivity=2)

    return torch.tensor(segmentation, dtype=torch.uint8)


def instance_segmentation_simple(semantic_map_uint8, contour_map_float32, size_threshold=10):
    """
    Using a semantic segmentation map and a contour map to separate touching objects and perform instance segmentation.\n
    Note this one use a simpler algorithm.

    Args:
        semantic_map_uint8 (np.Array): The input semantic segmented map.
        contour_map_float32 (np.Array): The input contour segmented map.
        size_threshold (int): The minimal size in pixel of each object. Object smaller than this will be removed.

    Returns:
        torch.Tensor: instance segmented map where 0 is background and every other value represent an object.
    """
    # Convert PyTorch tensors to NumPy arrays
    semantic_map_np = semantic_map_uint8.cpu().numpy().astype(np.int8)
    contour_map_np = contour_map_float32.cpu().numpy()

    # Convert contour map to int8
    contour_map_np = np.where(contour_map_np > 0.5, 1, 0).astype(np.int8)

    # Find boundary within the foreground object
    contour_map_np = contour_map_np * semantic_map_np

    new_map = np.clip(semantic_map_np - contour_map_np, 0, 1)

    structure = [
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    ]

    segmentation = label(new_map, structure=structure)

    # Remove small segments
    segmentation = morphology.remove_small_objects(segmentation[0], min_size=size_threshold, connectivity=2)

    return torch.tensor(segmentation, dtype=torch.uint8)

# 这里的函数用于测试各个组件是否能正常运作
if __name__ == "__main__":
    #fake_predictions = [torch.randn(4, 512, 512), torch.randn(4, 512, 512)]
    #predictions_to_final_img(fake_predictions)
    test_tensor = path_to_tensor("../test_img.tif")
    print(test_tensor)
    #print(test_loader.dataset[0][0])
    #test_tensor = path_to_tensor('Datasets/train/lab/Labels_Jul13ab_nntrain_1.tif', label=True)
    #print(test_tensor.shape)
    #test_tensor = composed_transform(test_tensor, 1)
    #print(test_tensor.shape)
    #print(test_tensor)
    #test_array = np.asarray(test_tensor)
    #im = imageio.volsave('Result.tif', test_array)
