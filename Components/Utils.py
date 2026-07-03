import gc
import json
import re
import math
import os
import torch
import imageio
import tifffile
import time
import h5py
import mrcfile

import numpy as np
import skimage.morphology as morph

from multiprocessing import shared_memory
from pathlib import Path
from scipy.ndimage import label
from torchvision.datasets.folder import has_file_allowed_extension

from . import Morphology as Morph, Augmentations as Aug
from .Welford import welford_mean_std


device = "cuda" if torch.cuda.is_available() else "cpu"


def multiple_loader(path, key):
    ext = os.path.splitext(path)[1].lower()

    if ext in ['.h5', '.hdf5', '.he5']:
        with h5py.File(path, 'r') as f:
            img = np.array(f[key])
    elif ext in ['.mrc']:
        with mrcfile.open(path, mode='r') as mrc:
            img = mrc.data
    else:  # All other formats
        img = imageio.v3.imread(path)

    return img

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
    img = multiple_loader(path, key)

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
        return img
    else:
        if norm_strategy == 'std':
            # Take less memory than Numpy's Existing method.
            mean, std = welford_mean_std(img)
            mean, std = np.array(mean).astype(np.float32), np.array(std).astype(np.float32)
            # Take less memory than img = (img - mean) / std
            out_img = np.empty_like(img, dtype=np.float16)
            for i in range(0, img.shape[0]):
                slice_img = img[i].astype(np.float32)
                slice_img -= mean
                np.divide(slice_img, std, slice_img)
                out_img[i] = slice_img
        '''elif norm_strategy == 'n1tp1':
            img = img.astype(np.float32, copy=False)
            q1, q99 = np.percentile(img[non_zero], [1, 99])
            # To 0 and 1
            img -= q1
            img /= (q99-q1)
            np.clip(img, 0, 1, img)
            # To -1 and 1
            img -= 0.5
            img *= 2'''
        return out_img

def path_to_array_nonorm(path, key='default'):
    img = multiple_loader(path, key)
    if len(img.shape) != 3:
        raise ValueError(f'Only 3D images are supported (Z, Y, X)! {path} seems to be {len(img.shape)}D! instead')
    mean, std = welford_mean_std(img)
    mean, std = np.array(mean).astype(np.float32), np.array(std).astype(np.float32)
    return img, mean, std


def make_label_pair_tv(image_dir, extensions=(".tif", ".tiff", ".mrc", ".h5", ".hdf5")):
    """
    Generate a list containing pairs of file paths to training images and their labels.
    The labels should have the same name as their corresponding image, with a prefix "Labels_".

    Args:
        image_dir (str): Path to the directory where images are stored.
        extensions: Acceptable image formats.

    Returns:
        Example:[('Datasets\\train\\testimg1.tif', 'Datasets\\train\\Labels_testimg1.tif'),
                 ('Datasets\\train\\testimg2.tif', 'Datasets\\train\\Labels_testimg2.tif')]
    """
    image_dir = Path(image_dir)
    image_label_pair = []
    image_files = os.listdir(image_dir)
    for fname in sorted(image_files):
        if has_file_allowed_extension(fname, extensions):
            if not "Labels_" in fname:
                path = image_dir.joinpath(fname)
                label_path = image_dir.joinpath('Labels_' + fname)
                image_label_pair.append((path, label_path))
    return image_label_pair


def make_path_list_predict(image_dir, extensions=(".tif", ".tiff", ".mrc", ".h5", ".hdf5")):
    """
    Generate a list containing file paths to images waiting to get predicted.

    Args:
        image_dir (str): Path to the directory where images are stored.
        extensions: Acceptable image formats.

    Returns:
        Example:['Datasets\\predict\\testpic1.tif',
                 'Datasets\\predict\\testpic2.tif']
    """
    image_dir = Path(image_dir)
    path_list = []
    image_dir = image_dir.expanduser()
    image_files = os.listdir(image_dir)
    for fname in sorted(image_files):
        if has_file_allowed_extension(fname, extensions):
            path = image_dir.joinpath(fname)
            path_list.append(path)
    return path_list


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
        super().__init__()
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
                if len(original_array) % 2 != 0:
                    original_array = original_array[:-1]  # drop last index to make even
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
                if len(supervised_array) % 2 != 0:
                    supervised_array = supervised_array[:-1]  # drop last index
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
        return 2 * (len(self.data)//2)


def custom_collate(batch):
    if len(batch) == 1:
        batch = [torch.unsqueeze(sample, dim=0) for sample in batch[0]]
        return batch
    positive_samples = batch[0]
    negative_samples = batch[1]
    batch = [torch.stack((a_i, b_i), dim=0) for a_i, b_i in zip(positive_samples, negative_samples)]
    return batch


def stitch_output_volume(indexed_files, dim_info, hw_size, depth_size, start_idx=0):
    depth, height, width, depth_multiplier, height_multiplier, width_multiplier, total_multiplier = dim_info
    result_volume = torch.zeros((depth,
                                 height,
                                 width), dtype=torch.float16)

    for i in range(total_multiplier):
        patch_path = indexed_files[start_idx + i][1]
        patch = torch.from_numpy(tifffile.imread(patch_path)).squeeze((0, 1))

        depth_idx, height_idx, width_idx = (i // (height_multiplier * width_multiplier)) % depth_multiplier, \
                                           (i // width_multiplier) % height_multiplier, \
                                           i % width_multiplier

        depth_start = math.floor((depth_size - (depth_size * depth_multiplier - depth) / (
                depth_multiplier - 1)) * depth_idx) if depth_multiplier > 1 else 0
        height_start = math.floor((hw_size - (hw_size * height_multiplier - height) / (
                height_multiplier - 1)) * height_idx) if height_multiplier > 1 else 0
        width_start = math.floor((hw_size - (hw_size * width_multiplier - width) / (
                width_multiplier - 1)) * width_idx) if width_multiplier > 1 else 0

        result_volume[depth_start:depth_start + patch.shape[0],
        height_start:height_start + patch.shape[1],
        width_start:width_start + patch.shape[2]] = patch
    return result_volume

def load_indexed_files_for_stitching(prefix):
    folder = Path("Datasets/prediction_cache/")  # Hardcoded for now
    # Find all files starting with the prefix (any extension)
    # Use glob pattern: prefix*
    all_files = list(folder.glob(f"{prefix}*"))
    # Filter out directories and keep only files (optionally check extension)
    files = [f for f in all_files if f.is_file()]
    # Extract batch index from filename: prefix_{index} (ignoring extension)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)(\.tiff?)?$")
    indexed_files = []
    for f in files:
        match = pattern.match(f.name)
        if match:
            idx = int(match.group(1))
            indexed_files.append((idx, f))
        else:
            print(f"Warning: file '{f.name}' does not match expected pattern, skipping.")

    # Sort by index and load arrays
    indexed_files.sort(key=lambda x: x[0])
    return indexed_files

def stitch_output_volumes(prefix, meta_list, hw_size=128, depth_size=128):
    """
    Stitch the patches of output volumes and reconstruct the original image tensor(s).

    Args:
        prefix (str): The filename prefix (e.g., "Pixels_", "Contour_", "Semantic_").
        meta_list (list): The meta information auto generated by Predict_Dataset class.
        hw_size (int): The height and width of the patches. In pixels.
        depth_size (int): The depth of the patches. In pixels.

    Returns:
        list: stitched tensors with shape (C, D, H, W).
    """
    output_list = []

    folder = Path("Datasets/prediction_cache/") # Hardcoded for now

    # Find all files starting with the prefix (any extension)
    # Use glob pattern: prefix*
    all_files = list(folder.glob(f"{prefix}*"))

    # Filter out directories and keep only files (optionally check extension)
    files = [f for f in all_files if f.is_file()]

    # Extract batch index from filename: prefix_{index} (ignoring extension)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)(\.tiff?)?$")
    indexed_files = []
    for f in files:
        match = pattern.match(f.name)
        if match:
            idx = int(match.group(1))
            indexed_files.append((idx, f))
        else:
            print(f"Warning: file '{f.name}' does not match expected pattern, skipping.")

    # Sort by index and load arrays
    indexed_files.sort(key=lambda x: x[0])
    current_offset = 0
    for meta_info in meta_list:
        out_file_name = "Mask_" + meta_info[0]
        depth, height, width = meta_info[1][1:]
        depth_multiplier = math.ceil(depth / depth_size)
        height_multiplier = math.ceil(height / hw_size)
        width_multiplier = math.ceil(width / hw_size)
        total_multiplier = depth_multiplier * height_multiplier * width_multiplier
        dim_info = depth, height, width, depth_multiplier, height_multiplier, width_multiplier, total_multiplier
        result_volume = stitch_output_volume(indexed_files, dim_info, hw_size, depth_size, current_offset)
        output_list.append((result_volume, out_file_name))
        current_offset += total_multiplier

    for _, file_path in indexed_files:
        try:
            file_path.unlink()
        except OSError as e:
            print(f"Warning: Could not delete {file_path}: {e}")
    return output_list


def predictions_to_final_img(stitched_volume, direc):
    """
    Save the stitched patches into the selected directory.\n
    This one is for semantic segmentation.

    Args:
        direc (str): Directory where the final images will be saved to.
    """
    array = stitched_volume[0].numpy()
    array = np.where(array >= 0.5, np.uint8(1), np.uint8(0))
    imageio.v3.imwrite(uri=f'{direc}/{stitched_volume[1]}', image=array)

def tiff_size_estimate(tensor: np.ndarray):
    threshold = 3.95*(1024**3)
    nbytes = tensor.nbytes
    if nbytes > threshold:
        print('Detected an segmentation that will exceed 4GB when saved! Saving as BigTIFF format! Which not all image readers can open!')
        return True
    else:
        return False


def predictions_to_final_img_instance(stitched_volume_p, stitched_volume_c, direc, segmentation_mode='simple', dynamic=10, pixel_reclaim=True):
    """
    Save the switched patches into the selected directory.\n
    This one is for instance segmentation.

    Args:
        direc (str): Directory where the final images will be saved to.
        segmentation_mode (str): If 'simple', will identify objects via simple connected component labelling. If 'watershed', will use a distance transform watershed instead, which is slower but yield much less under-segment.
        dynamic (int): Dynamic of intensity for the search of regional minima in the distance transform image. Increasing its value will yield more object merges. Default: 10.
        pixel_reclaim (bool): Whether to reclaim lost pixel during the instance segmentation, a slow process. Default: True
    """
    tifffile.imwrite(f'{direc}/Pixels_{stitched_volume_p[1]}',
                     data=stitched_volume_p[0].numpy(), bigtiff=tiff_size_estimate(stitched_volume_p[0].numpy()))
    tifffile.imwrite(f'{direc}/Contour_{stitched_volume_c[1]}',
                     data=stitched_volume_c[0].numpy(), bigtiff=tiff_size_estimate(stitched_volume_c[0].numpy()))
    print(f'Computing instance segmentation using contour data for {stitched_volume_c[1]}... Can take a while if the image is big.')
    instance_array = instance_segmentation_simple(stitched_volume_p[0], stitched_volume_c[0],
                                                  mode=segmentation_mode, dynamic=dynamic, pixel_reclaim=pixel_reclaim)
    tifffile.imwrite(f'{direc}/Instance_{stitched_volume_c[1]}',
                     data=instance_array, bigtiff=tiff_size_estimate(instance_array))


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

def perform_watershed(segmentation, dynamic):
    """
    Perform watershed segmentation on the pre‑segmented mask.
    """
    distance_map = Morph.chamfer_distance_transform_parallel(segmentation, dtype=np.uint16, num_core=min(max((os.cpu_count()//2)-1, 1), 32))
    distance_map = Morph.inverter(distance_map)

    hmin = Morph.geodesic_reconstruction_by_erosion(distance_map, dynamic)
    #hmin = morph.local_minima(hmin).astype(np.uint16)
    #label(hmin, output=hmin)
    hmin = Morph.label_h_minima(hmin, distance_map, dynamic)
    segmentation = Morph.marker_controlled_watershed(distance_map, markers=hmin, mask=segmentation)
    return segmentation

def instance_segmentation_simple(semantic_map, contour_map, size_threshold=10, mode='simple', dynamic=10, pixel_reclaim=True, distance_threshold=2):
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
        distance_threshold (int): The radius in pixels to search for nearby pixels when allocating. Default: 2.

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
    segmentation = segmentation.numpy().astype(np.uint16)
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
        segmentation = perform_watershed(segmentation, dynamic)

    morph.remove_small_objects(segmentation, min_size=size_threshold, connectivity=structure, out=segmentation)
    del structure

    if pixel_reclaim:
        start_time = time.time()
        segmentation = Morph.pixel_reclaim(touching_map.numpy(), segmentation, distance_threshold)
        del touching_map
        elapsed_time = time.time() - start_time
        print(f"Time taken for pixel reclaim: {elapsed_time}")
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


def load_tiff_stack(folder_path: str, prefix: str) -> torch.Tensor:
    """
    Load all TIFF files in `folder_path` whose names start with `prefix`,
    stack them into a single PyTorch tensor, and return it.

    Files are assumed to have names like `prefix_{batch_idx}` (optionally with
    .tif or .tiff extension). They are sorted by `batch_idx` before stacking.
    """
    folder = Path(folder_path)

    # Find all files starting with the prefix (any extension)
    # Use glob pattern: prefix*
    all_files = list(folder.glob(f"{prefix}*"))

    # Filter out directories and keep only files (optionally check extension)
    files = [f for f in all_files if f.is_file()]

    # Extract batch index from filename: prefix_{index} (ignoring extension)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)(\.tiff?)?$")
    indexed_files = []
    for f in files:
        match = pattern.match(f.name)
        if match:
            idx = int(match.group(1))
            indexed_files.append((idx, f))
        else:
            # If the filename doesn't match the pattern, skip or you may warn
            # For robustness, we skip such files; you can change this behavior.
            print(f"Warning: file '{f.name}' does not match expected pattern, skipping.")

    if not indexed_files:
        raise ValueError(f"No files matched the expected pattern '{prefix}<integer>' in {folder_path}")

    # Sort by index and load arrays
    indexed_files.sort(key=lambda x: x[0])
    # Load the first file to determine shape and dtype
    first_idx, first_path = indexed_files[0]
    first_arr = tifffile.imread(first_path)
    # Pre‑allocate the output tensor
    num_files = len(indexed_files)
    out_tensor = torch.empty((num_files, *first_arr.shape), dtype=torch.from_numpy(first_arr).dtype)
    out_tensor[0] = torch.from_numpy(first_arr)

    # Load the remaining files and fill the tensor
    for i, (idx, file_path) in enumerate(indexed_files[1:], start=1):
        arr = tifffile.imread(file_path)
        out_tensor[i] = torch.from_numpy(arr)

    # Delete all loaded files
    for _, file_path in indexed_files:
        try:
            file_path.unlink()
        except OSError as e:
            print(f"Warning: Could not delete {file_path}: {e}")

    return out_tensor


def calculate_val_start_end(multiplier, required_size, full_size, idx):
    if multiplier > 1:
        start = (required_size - ((required_size * multiplier - full_size) / (multiplier - 1))) * idx
        start = math.floor(start)
    else:
        start = 0
    end = start + required_size
    return start, end


def calculate_predict_start_end(multiplier, required_size, full_size, idx, required_overlap):
    if multiplier > 1:
        start = (required_size - ((required_size * multiplier - full_size) / (multiplier - 1))) * idx
        start = math.floor(start)
    else:
        start = 0
    end = start + required_size + (2 * required_overlap)
    return start, end


def get_contour_maps(label_file_path, folder_path='generated_contour_maps', contour_map_width=1):
    """
    Try get the contour map for the label file. Will first try to load previously saved one from folder_path,
    if failed, will generate new one and save it to folder_path.
    """
    img_name = label_file_path.name
    contour_img_name = "Contour_" + img_name
    contour_img_path = os.path.join(folder_path, contour_img_name)
    metadata_path = os.path.join(folder_path, "contour_metadata.json")

    # Use file modification time and file size as a verification method
    stat = label_file_path.stat()
    current_sig = f"mtime:{stat.st_mtime_ns}|size:{stat.st_size}"

    # Load existing metadata
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}

    # Check if regeneration is needed
    needs_regenerate = True

    if os.path.exists(contour_img_path):
        # Check if we have metadata for this file
        if str(label_file_path) in metadata:
            # Compare current signature with stored signature
            if metadata[str(label_file_path)] == current_sig:
                needs_regenerate = False

    if needs_regenerate:
        print(f'Generating contour map for {img_name}... Can take a while if there are lots of objects.')
        contour_img = Aug.instance_contour_transform(path_to_array(str(label_file_path), label=True), contour_outward=contour_map_width)
        imageio.v3.imwrite(uri=f'{contour_img_path}', image=contour_img)
        metadata[str(label_file_path)] = current_sig
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f'Saved {contour_img_name}')
    else:
        contour_img = path_to_array(contour_img_path, label=True)
        print(f'Loaded previously saved {contour_img_name}')
    return contour_img
