import json
import math
import os
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

import zarr

from Components.Augmentations import apply_aug, apply_aug_unsupervised, binarisation
from Components.Utils import path_to_array, path_to_array_nonorm, make_label_pair_tv, make_path_list_predict, \
    calculate_val_start_end, calculate_predict_start_end, get_contour_maps, compute_eligibility_masks


def load_metadata(metadata_file_path):
    """Load existing metadata or return empty dict"""
    if metadata_file_path.exists():
        with open(metadata_file_path, 'r') as f:
            return json.load(f)
    return {}


def save_metadata(metadata_file_path, source_metadata):
    """Save current metadata to file"""
    with open(metadata_file_path, 'w') as f:
        json.dump(source_metadata, f, indent=2)


def get_file_signature(file_path):
    stat = file_path.stat()
    # Use file modification time and file size as a verification method
    return f"mtime:{stat.st_mtime_ns}|size:{stat.st_size}"


def needs_reprocessing(source_metadata, source_path, zarr_path, required_attrs=None):
    if not Path(source_path).exists():
        return True
    if not zarr_path.exists():
        return True
    if source_metadata.get(str(source_path)) != get_file_signature(source_path):
        return True
    if required_attrs:
        arr = zarr.open(zarr_path, mode='r')  # cheap: only reads .zattrs
        for attr in required_attrs:
            if attr not in arr.attrs:
                return True
    return False


def safe_remove(path):
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception as e:
        pass
        #print(f"Error removing {path}: {e}")


def process_file(preprocessed_dir, source_metadata, hdf5_key, chunk_size,
                 source_path, is_label, processing_func=None):
    zarr.config.set({'async.concurrency': 2, 'async.timeout': None})
    base = source_path.stem
    zarr_path = preprocessed_dir.joinpath(f"{base}.zarr")

    required_attrs = ['fg_proportion'] if is_label else ['mean', 'std']
    if needs_reprocessing(source_metadata, source_path, zarr_path, required_attrs):
        print(f"Converting {base} into zarr format...")
        safe_remove(zarr_path)
        if is_label:
            vol = path_to_array(str(source_path), True, hdf5_key)[None, :]
        else:
            vol, mean, std = path_to_array_nonorm(str(source_path), hdf5_key)
            vol = vol[None, :]
        if processing_func:
            vol = processing_func(vol)

        store = zarr.storage.LocalStore(zarr_path)
        vol_arr = zarr.create_array(store=store, data=vol, chunks=chunk_size)
        if is_label:
            # computed once during conversion — no per-epoch loading
            vol_arr.attrs['fg_proportion'] = float(vol.mean())
        else:
            vol_arr.attrs['mean'] = float(mean)
            vol_arr.attrs['std'] = float(std)
        source_metadata[str(source_path)] = get_file_signature(source_path)

    return zarr_path


class TrainDatasetChunked(torch.utils.data.Dataset):
    def __init__(self, images_dir, augmentation_csv, hw_size=64, d_size=64,
                 instance_mode=False, contour_map_width=1, hdf5_key='Default'):
        self.file_list = np.array(make_label_pair_tv(images_dir))
        self.num_files = len(self.file_list)
        self.preprocessed_dir = Path(os.path.join(images_dir, "zarr_preprocessed"))
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Metadata file for verification
        self.metadata_file_path = self.preprocessed_dir.joinpath("source_metadata.json")
        self.source_metadata = load_metadata(self.metadata_file_path)

        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.hw_size = hw_size
        self.d_size = d_size
        self.instance_mode = instance_mode
        self.contour_map_width = contour_map_width
        self.hdf5_key = hdf5_key

        chunk_depth = 2 ** math.floor(math.log2(d_size) + 1)
        chunk_hw = 2 ** math.floor(math.log2(hw_size) + 1)
        self.chunk_size = (1, chunk_depth, chunk_hw, chunk_hw)

        self.zarr_paths, fg_proportions = self.preprocess_files()

        # zarr .shape is metadata-only; no chunk data is read here
        log_size = [math.log(math.prod(zarr.open(p[0], mode='r').shape))
                    for p in self.zarr_paths]

        masks = compute_eligibility_masks(fg_proportions, log_size)
        self.fg_proportions    = masks["fg_proportions"]
        self.eligible_positive = masks["eligible_positive"]
        self.positive_log_size = masks["positive_log_size"]
        self.eligible_negative = masks["eligible_negative"]
        self.negative_log_size = masks["negative_log_size"]
        self.eligible_mixed    = masks["eligible_mixed"]
        self.mixed_log_size    = masks["mixed_log_size"]

    def get_statistics(self):
        return {
            "num_files": int(self.num_files),
            "fg_proportions": list(self.fg_proportions),
            "num_eligible_positive": int(len(self.eligible_positive)),
            "num_eligible_negative": int(len(self.eligible_negative)),
            "num_eligible_mixed": int(len(self.eligible_mixed)),
        }

    def preprocess_files(self):
        zarr_paths = []
        fg_proportions = []
        for img_path, lab_path in self.file_list:
            img_zarr = process_file(self.preprocessed_dir, self.source_metadata,
                                    self.hdf5_key, self.chunk_size, img_path, False)
            lab_zarr = process_file(self.preprocessed_dir, self.source_metadata,
                                    self.hdf5_key, self.chunk_size, lab_path, True, binarisation)

            if self.instance_mode:
                contour_zarr = self.preprocessed_dir.joinpath(
                    f"Contour_{Path(img_zarr).stem}.zarr")
                if needs_reprocessing(self.source_metadata, lab_path, contour_zarr,
                                      required_attrs=['fg_proportion']):
                    print(f"Processing contour: {lab_path}")
                    safe_remove(contour_zarr)
                    vol = get_contour_maps(lab_path, 'generated_contour_maps',
                                           self.contour_map_width)[None, :]
                    store = zarr.storage.LocalStore(contour_zarr)
                    contour_arr = zarr.create_array(store=store, data=vol,
                                                    chunks=self.chunk_size)
                    contour_arr.attrs['fg_proportion'] = float(vol.mean())
                fg_proportions.append(
                    float(zarr.open(contour_zarr, mode='r').attrs['fg_proportion']))
                zarr_paths.append((img_zarr, lab_zarr, contour_zarr))
            else:
                fg_proportions.append(
                    float(zarr.open(lab_zarr, mode='r').attrs['fg_proportion']))
                zarr_paths.append((img_zarr, lab_zarr))

        save_metadata(self.metadata_file_path, self.source_metadata)
        return zarr_paths, fg_proportions

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        negative_control = idx[1]
        idx = idx[0]
        background_threshold = 0.01
        foreground_threshold = 0.01
        if negative_control == 'positive':
            foreground_threshold = 0.01
            background_threshold = 0
        elif negative_control == 'negative':
            background_threshold = 0.01
            foreground_threshold = 0

        if self.instance_mode:
            img_path, lab_path, contour_path = self.zarr_paths[idx]
            contour_vol = zarr.open(contour_path, mode='r')
        else:
            img_path, lab_path = self.zarr_paths[idx]

        img_vol, lab_vol = zarr.open(img_path, mode='r'), zarr.open(lab_path, mode='r')
        img_mean, img_std = img_vol.attrs['mean'], img_vol.attrs['std']
        if self.instance_mode:
            img_tensor, lab_tensor, contour_tensor = apply_aug(img_vol, lab_vol, contour_vol,
                                                               self.augmentation_params, self.hw_size, self.d_size,
                                                               foreground_threshold, background_threshold, True, img_mean, img_std)
            return img_tensor, lab_tensor, contour_tensor
        else:
            img_tensor, lab_tensor = apply_aug(img_vol, lab_vol, None,
                                               self.augmentation_params, self.hw_size, self.d_size,
                                               foreground_threshold, background_threshold, True, img_mean, img_std)
            return img_tensor, lab_tensor


class UnsupervisedDatasetChunked(torch.utils.data.Dataset):
    def __init__(self, images_dir, augmentation_csv, hw_size=64, d_size=64, hdf5_key='Default'):
        self.file_list = np.array(make_path_list_predict(images_dir))
        self.num_files = len(self.file_list)
        self.preprocessed_dir = Path(os.path.join(images_dir, "zarr_preprocessed"))
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Metadata file for verification
        self.metadata_file_path = self.preprocessed_dir.joinpath("source_metadata.json")
        self.source_metadata = load_metadata(self.metadata_file_path)

        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.hw_size = hw_size
        self.d_size = d_size
        self.hdf5_key = hdf5_key

        chunk_depth = 2 ** math.floor(math.log2(d_size) + 1)
        chunk_hw = 2 ** math.floor(math.log2(hw_size) + 1)
        self.chunk_size = (1, chunk_depth, chunk_hw, chunk_hw)

        self.zarr_paths = self.preprocess_files()
        self.log_size = [math.log(math.prod(zarr.open(zp, mode='r').shape))
                         for zp in self.zarr_paths]

    def __len__(self):
        return self.num_files

    def preprocess_files(self):
        zarr_paths = []
        for img_path in self.file_list:
            img_zarr = process_file(self.preprocessed_dir, self.source_metadata, self.hdf5_key, self.chunk_size, img_path, False)
            zarr_paths.append(img_zarr)

        save_metadata(self.metadata_file_path, self.source_metadata)
        return zarr_paths

    def __getitem__(self, idx):
        img_vol = zarr.open(self.zarr_paths[idx], mode='r')
        img_mean, img_std = img_vol.attrs['mean'], img_vol.attrs['std']
        img_tensor = apply_aug_unsupervised(img_vol, self.augmentation_params, self.hw_size, self.d_size, True, img_mean, img_std)
        return (img_tensor,)


class ValDatasetChunked(torch.utils.data.Dataset):
    def __init__(self, images_dir, hw_size, d_size, instance_mode, contour_map_width=1, hdf5_key='Default'):
        self.file_list = np.array(make_label_pair_tv(images_dir))
        self.hw_size = hw_size
        self.d_size = d_size
        self.num_files = len(self.file_list)
        self.instance_mode = instance_mode
        self.contour_map_width = contour_map_width
        self.hdf5_key = hdf5_key

        self.preprocessed_dir = Path(os.path.join(images_dir, "tiff_preprocessed"))
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Metadata file for verification
        self.metadata_file_path = self.preprocessed_dir / "source_metadata.json"
        self.source_metadata = load_metadata(self.metadata_file_path)

        self.patch_paths = []  # Stores tuples of (img_path, lab_path, [contour_path])
        self.original_shapes = []  # Stores original shapes for each file
        self.total_patches = 0

        # Process each image-label pair
        for img_path, lab_path in self.file_list:
            self.process_image_pair(Path(img_path), Path(lab_path))

        save_metadata(self.metadata_file_path, self.source_metadata)
        super().__init__()

    def process_image_pair(self, img_path, lab_path):
        # Create unique identifier for this pair
        pair_id = f"{img_path.stem}"
        pair_dir = self.preprocessed_dir.joinpath(pair_id)
        os.makedirs(pair_dir, exist_ok=True)

        # Get current signatures
        img_sig = get_file_signature(img_path)
        lab_sig = get_file_signature(lab_path)

        # Check if reprocessing is needed
        needs_processing = True
        if pair_id in self.source_metadata:
            entry = self.source_metadata[pair_id]
            if (entry['img_signature'] == img_sig and
                entry['lab_signature'] == lab_sig and
                entry['hw_size'] == self.hw_size and
                entry['d_size'] == self.d_size and
                pair_dir.exists()):
                # Verify all patch files exist
                all_patches_exist = True
                for patch_entry in entry['patches']:
                    if not Path(patch_entry['img_path']).exists():
                        all_patches_exist = False
                    if not Path(patch_entry['lab_path']).exists():
                        all_patches_exist = False
                    if self.instance_mode:
                        if not 'contour_path' in patch_entry:
                            all_patches_exist = False
                        elif not Path(patch_entry['contour_path']).exists():
                            all_patches_exist = False

                if all_patches_exist:
                    self.patch_paths.extend(entry['patches'])
                    self.original_shapes.append(entry['original_shape'])
                    self.total_patches += len(entry['patches'])
                    needs_processing = False

        if needs_processing:
            print(f"Processing validation pair: {pair_id}")
            safe_remove(pair_dir)
            os.makedirs(pair_dir, exist_ok=True)

            # Load original arrays
            img_array = path_to_array(str(img_path), key=self.hdf5_key, label=False)
            lab_array = binarisation(path_to_array(str(lab_path), label=True))

            # Get contour maps if needed
            contour_array = None
            if self.instance_mode:
                contour_array = get_contour_maps(lab_path, 'generated_contour_maps', self.contour_map_width)

            depth, height, width = img_array.shape
            original_shape = (depth, height, width)

            # Calculate number of patches
            depth_multiplier = math.ceil(depth / self.d_size)
            height_multiplier = math.ceil(height / self.hw_size)
            width_multiplier = math.ceil(width / self.hw_size)

            patch_data = []
            # Generate and save patches
            for depth_idx, height_idx, width_idx in product(range(depth_multiplier),
                                                            range(height_multiplier),
                                                            range(width_multiplier)):
                depth_start, depth_end = calculate_val_start_end(depth_multiplier, self.d_size, depth, depth_idx)
                height_start, height_end = calculate_val_start_end(height_multiplier, self.hw_size, height, height_idx)
                width_start, width_end = calculate_val_start_end(width_multiplier, self.hw_size, width, width_idx)

                # Extract patches
                img_patch = img_array[depth_start:depth_end,
                            height_start:height_end,
                            width_start:width_end]

                lab_patch = lab_array[depth_start:depth_end,
                            height_start:height_end,
                            width_start:width_end]

                pad_d = max(0, self.d_size - img_patch.shape[0])
                pad_h = max(0, self.hw_size - img_patch.shape[1])
                pad_w = max(0, self.hw_size - img_patch.shape[2])
                if pad_d or pad_h or pad_w:
                    pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
                    img_patch = np.pad(img_patch, pad_width, mode='constant')
                    lab_patch = np.pad(lab_patch, pad_width, mode='constant')

                # Save patches
                patch_id = f"patch_{depth_idx}_{height_idx}_{width_idx}"
                img_patch_path = pair_dir / f"{patch_id}_img.tiff"
                lab_patch_path = pair_dir / f"{patch_id}_lab.tiff"

                tifffile.imwrite(str(img_patch_path), img_patch, compression='zstd')
                tifffile.imwrite(str(lab_patch_path), lab_patch, compression='zstd')

                patch_entry = {
                    'img_path': str(img_patch_path),
                    'lab_path': str(lab_patch_path),
                }

                # Process contour if needed
                if self.instance_mode:
                    contour_patch = contour_array[depth_start:depth_end,
                                    height_start:height_end,
                                    width_start:width_end]
                    contour_patch_path = pair_dir / f"{patch_id}_contour.tiff"
                    tifffile.imwrite(str(contour_patch_path), contour_patch, compression='zstd')
                    patch_entry['contour_path'] = str(contour_patch_path)

                patch_data.append(patch_entry)

            # Update metadata
            self.source_metadata[pair_id] = {
                'img_signature': img_sig,
                'lab_signature': lab_sig,
                'hw_size': self.hw_size,
                'd_size': self.d_size,
                'original_shape': original_shape,
                'patches': patch_data
            }

            self.patch_paths.extend(patch_data)
            self.original_shapes.append(original_shape)
            self.total_patches += len(patch_data)

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        patch_entry = self.patch_paths[idx]
        img_patch = tifffile.imread(patch_entry['img_path'])
        lab_patch = tifffile.imread(patch_entry['lab_path'])

        img_tensor = torch.from_numpy(img_patch[None, :])  # Add channel dimension
        lab_tensor = torch.from_numpy(lab_patch[None, :])

        if self.instance_mode:
            contour_patch = tifffile.imread(patch_entry['contour_path'])
            contour_tensor = torch.from_numpy(contour_patch[None, :])
            return img_tensor.to(torch.float32), lab_tensor.to(torch.float32), contour_tensor.to(torch.float32)
        else:
            return img_tensor.to(torch.float32), lab_tensor.to(torch.float32)


class PredictDatasetChunked(torch.utils.data.Dataset):
    def __init__(self, images_dir, hw_size=128, depth_size=128, hw_overlap=16, depth_overlap=16, hdf5_key='Default'):
        self.file_list = make_path_list_predict(images_dir)
        self.preprocessed_dir = Path(os.path.join(images_dir, "tiff_preprocessed"))
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Metadata file for verification
        self.metadata_file_path = self.preprocessed_dir.joinpath("source_metadata.json")
        self.source_metadata = load_metadata(self.metadata_file_path)

        self.hw_size = hw_size
        self.depth_size = depth_size
        self.hw_overlap = hw_overlap
        self.depth_overlap = depth_overlap
        self.hdf5_key = hdf5_key

        self.patch_paths = []  # Stores paths to all patch TIFFs
        self.meta_list = []  # Stores original file metadata
        self.original_shapes = []  # Stores original shapes for each file

        # Process each image into patches
        for file_path in self.file_list:
            self.process_image_into_patches(Path(file_path))

        save_metadata(self.metadata_file_path, self.source_metadata)

    def process_image_into_patches(self, source_path):
        current_signature = get_file_signature(source_path)
        base      = source_path.stem
        image_dir = self.preprocessed_dir.joinpath(base)
        os.makedirs(image_dir, exist_ok=True)

        source_sig       = get_file_signature(source_path)
        needs_processing = True
        if source_path in self.source_metadata:
            entry = self.source_metadata[source_path]
            if (entry['signature']   == source_sig and
                entry['hw_size']     == self.hw_size and
                entry['depth_size']  == self.depth_size):
                if all(Path(p).exists() for p in entry['patch_paths']):
                    needs_processing = False

        if needs_processing:
            print(f"Processing image: {source_path.name}")
            safe_remove(image_dir); os.makedirs(image_dir, exist_ok=True)

            image = path_to_array(str(source_path), key=self.hdf5_key, label=False)[None, :]   # (1, D, H, W)
            c, depth, height, width = image.shape

            depth_multiplier  = math.ceil(depth  / self.depth_size)
            height_multiplier = math.ceil(height / self.hw_size)
            width_multiplier  = math.ceil(width  / self.hw_size)

            paddings = ((0, 0),
                        (self.depth_overlap, self.depth_overlap),
                        (self.hw_overlap,    self.hw_overlap),
                        (self.hw_overlap,    self.hw_overlap))
            padded_image = np.pad(image, paddings, mode="symmetric")

            expected_d = self.depth_size + 2 * self.depth_overlap
            expected_h = self.hw_size    + 2 * self.hw_overlap
            expected_w = self.hw_size    + 2 * self.hw_overlap

            patch_paths = []
            for depth_idx, height_idx, width_idx in product(range(depth_multiplier),
                                                            range(height_multiplier),
                                                            range(width_multiplier)):
                depth_start,  depth_end  = calculate_predict_start_end(depth_multiplier,  self.depth_size, depth,  depth_idx,  self.depth_overlap)
                height_start, height_end = calculate_predict_start_end(height_multiplier, self.hw_size,    height, height_idx, self.hw_overlap)
                width_start,  width_end  = calculate_predict_start_end(width_multiplier,  self.hw_size,    width,  width_idx,  self.hw_overlap)

                patch = padded_image[:,
                                     depth_start:depth_end,
                                     height_start:height_end,
                                     width_start:width_end]

                # ---- NEW: pad chunk to expected patch size ----
                pad_d = max(0, expected_d - patch.shape[1])
                pad_h = max(0, expected_h - patch.shape[2])
                pad_w = max(0, expected_w - patch.shape[3])
                if pad_d or pad_h or pad_w:
                    patch = np.pad(patch,
                                   ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                                   mode='constant')

                patch_path = image_dir / f"patch_{depth_idx}_{height_idx}_{width_idx}.tiff"
                tifffile.imwrite(str(patch_path),
                                 np.squeeze(patch, axis=0),
                                 compression='zstd',
                                 compressionargs={'level': 3})
                patch_paths.append(str(patch_path))

            self.source_metadata[str(source_path)] = {
                'signature':       current_signature,
                'original_shape':  list(image.shape),
                'patch_paths':     patch_paths,
                'hw_size':         self.hw_size,
                'depth_size':      self.depth_size,
                'padding_version': 1,                          # <-- NEW
            }
            self.original_shapes.append(image.shape)
            self.patch_paths.extend(patch_paths)
        else:
            entry = self.source_metadata[str(source_path)]
            self.patch_paths.extend(entry['patch_paths'])
            self.original_shapes.append(entry['original_shape'])

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        patch = tifffile.imread(patch_path)
        return torch.from_numpy(patch[None, :]).to(torch.float32), self.hw_overlap, self.depth_overlap  # Add channel dimension

    def __getmetainfo__(self):
        # Return metadata for each original file
        return [(Path(path).name, tuple(shape))
                for path, shape in zip(self.file_list, self.original_shapes)]


