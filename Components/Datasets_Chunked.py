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
    calculate_val_start_end, calculate_predict_start_end, get_contour_maps


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


def needs_reprocessing(source_metadata, source_path, zarr_path):
    if not Path(source_path).exists():
        return True
    if not zarr_path.exists():
        return True
    return source_metadata.get(str(source_path)) != get_file_signature(source_path)


def safe_remove(path):
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception as e:
        pass
        #print(f"Error removing {path}: {e}")


def process_file(preprocessed_dir, source_metadata, hdf5_key, chunk_size, source_path, is_label, processing_func=None):
    zarr.config.set({'async.concurrency': 2, 'async.timeout': None})
    base = source_path.stem
    zarr_path = preprocessed_dir.joinpath(f"{base}.zarr")

    if needs_reprocessing(source_metadata, source_path, zarr_path):
        print(f"Converting {base} into zarr format...")
        safe_remove(zarr_path)
        if is_label:
            vol = path_to_array(str(source_path), True, hdf5_key)[None, :]
        else:
            vol, mean, std = path_to_array_nonorm(str(source_path), hdf5_key)
            vol = vol[None, :]
        if processing_func:
            vol = processing_func(vol)

        if is_label:
            zarr.save_array(zarr_path, vol, chunk_shape=chunk_size)
        else:
            store = zarr.storage.LocalStore(zarr_path)
            # Store the arrays with appropriate chunks
            vol = zarr.create_array(store=store, data=vol, chunks=chunk_size)
            vol.attrs['mean'] = float(mean)
            vol.attrs['std'] = float(std)
        source_metadata[str(source_path)] = get_file_signature(source_path)

    return zarr_path


class TrainDatasetChunked(torch.utils.data.Dataset):
    def __init__(self, images_dir, augmentation_csv, train_multiplier=1, hw_size=64, d_size=64,
                 instance_mode=False, contour_map_width=1, hdf5_key='Default'):
        self.file_list = np.array(make_label_pair_tv(images_dir))
        self.num_files = len(self.file_list)
        self.preprocessed_dir = Path(os.path.join(images_dir, "zarr_preprocessed"))
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Metadata file for verification
        self.metadata_file_path = self.preprocessed_dir.joinpath("source_metadata.json")
        self.source_metadata = load_metadata(self.metadata_file_path)

        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.train_multiplier = train_multiplier
        self.hw_size = hw_size
        self.d_size = d_size
        self.instance_mode = instance_mode
        self.contour_map_width = contour_map_width
        self.hdf5_key = hdf5_key

        chunk_depth = 2 ** math.floor(math.log2(d_size) + 1)
        chunk_hw = 2 ** math.floor(math.log2(hw_size) + 1)
        self.chunk_size = (1, chunk_depth, chunk_hw, chunk_hw)

        self.zarr_paths = self.preprocess_files()

    def preprocess_files(self):
        zarr_paths = []
        for img_path, lab_path in self.file_list:
            img_zarr = process_file(self.preprocessed_dir, self.source_metadata, self.hdf5_key, self.chunk_size, img_path, False)
            lab_zarr = process_file(self.preprocessed_dir, self.source_metadata, self.hdf5_key, self.chunk_size, lab_path, True, binarisation)

            if self.instance_mode:
                contour_zarr = self.preprocessed_dir.joinpath(f"Contour_{Path(img_zarr).stem}.zarr")
                if needs_reprocessing(self.source_metadata, lab_path, contour_zarr):
                    print(f"Processing contour: {lab_path}")
                    safe_remove(contour_zarr)
                    vol = get_contour_maps(lab_path, 'generated_contour_maps', self.contour_map_width)[None, :]
                    zarr.save_array(contour_zarr, vol, chunk_shape=self.chunk_size)
                zarr_paths.append((img_zarr, lab_zarr, contour_zarr))
            else:
                zarr_paths.append((img_zarr, lab_zarr))

        save_metadata(self.metadata_file_path, self.source_metadata)
        return zarr_paths

    def __len__(self):
        return self.num_files * self.train_multiplier

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
        idx = math.floor(idx / self.train_multiplier)

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
    def __init__(self, images_dir, augmentation_csv, train_multiplier=1, hw_size=64, d_size=64, hdf5_key='Default'):
        self.file_list = np.array(make_path_list_predict(images_dir))
        self.num_files = len(self.file_list)
        self.preprocessed_dir = Path(os.path.join(images_dir, "zarr_preprocessed"))
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Metadata file for verification
        self.metadata_file_path = self.preprocessed_dir.joinpath("source_metadata.json")
        self.source_metadata = load_metadata(self.metadata_file_path)

        self.augmentation_params = pd.read_csv(augmentation_csv)
        self.train_multiplier = train_multiplier
        self.hw_size = hw_size
        self.d_size = d_size
        self.hdf5_key = hdf5_key

        chunk_depth = 2 ** math.floor(math.log2(d_size) + 1)
        chunk_hw = 2 ** math.floor(math.log2(hw_size) + 1)
        self.chunk_size = (1, chunk_depth, chunk_hw, chunk_hw)

        self.zarr_paths = self.preprocess_files()

    def __len__(self):
        return self.num_files * self.train_multiplier

    def preprocess_files(self):
        zarr_paths = []
        for img_path in self.file_list:
            img_zarr = process_file(self.preprocessed_dir, self.source_metadata, self.hdf5_key, self.chunk_size, img_path, False)
            zarr_paths.append(img_zarr)

        save_metadata(self.metadata_file_path, self.source_metadata)
        return zarr_paths

    def __getitem__(self, idx):
        idx = math.floor(idx / self.train_multiplier)
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
        base = source_path.stem
        image_dir = self.preprocessed_dir.joinpath(base)
        os.makedirs(image_dir, exist_ok=True)

        # Check if reprocessing is needed
        source_sig = get_file_signature(source_path)
        needs_processing = True
        if source_path in self.source_metadata:
            entry = self.source_metadata[source_path]
            if (entry['signature'] == source_sig and
                entry['hw_size'] == self.hw_size and
                entry['depth_size'] == self.depth_size):
                # Verify all patch files exist
                all_patches_exist = True
                for patch_entry in entry['patch_paths']:
                    if not Path(patch_entry).exists():
                        all_patches_exist = False

                if all_patches_exist:
                    needs_processing = False

        # Check if reprocessing is needed
        if needs_processing:
            print(f"Processing image: {source_path.name}")
            safe_remove(image_dir)
            os.makedirs(image_dir, exist_ok=True)

            # Load and pad the image
            image = path_to_array(str(source_path), key=self.hdf5_key, label=False)[None, :]  # [1, D, H, W]
            c, depth, height, width = image.shape

            # Calculate number of patches
            depth_multiplier = math.ceil(depth / self.depth_size)
            height_multiplier = math.ceil(height / self.hw_size)
            width_multiplier = math.ceil(width / self.hw_size)

            # Apply symmetric padding
            paddings = (
                (0, 0),
                (self.depth_overlap, self.depth_overlap),
                (self.hw_overlap, self.hw_overlap),
                (self.hw_overlap, self.hw_overlap)
            )
            padded_image = np.pad(image, paddings, mode="symmetric")

            # Store patch paths for this image
            patch_paths = []

            # Generate patches
            for depth_idx, height_idx, width_idx in product(range(depth_multiplier),
                                                            range(height_multiplier),
                                                            range(width_multiplier)):
                depth_start, depth_end = calculate_predict_start_end(depth_multiplier, self.depth_size, depth, depth_idx,
                                                                     self.depth_overlap)
                height_start, height_end = calculate_predict_start_end(height_multiplier, self.hw_size, height, height_idx,
                                                                       self.hw_overlap)
                width_start, width_end = calculate_predict_start_end(width_multiplier, self.hw_size, width, width_idx,
                                                                     self.hw_overlap)

                # Extract patch
                patch = padded_image[
                        :,
                        depth_start:depth_end,
                        height_start:height_end,
                        width_start:width_end
                        ]
                # Save patch as TIFF
                patch_path = image_dir / f"patch_{depth_idx}_{height_idx}_{width_idx}.tiff"
                tifffile.imwrite(str(patch_path), np.squeeze(patch, axis=0), compression='zstd', compressionargs={'level': 3})
                patch_paths.append(str(patch_path))

            # Update metadata
            self.source_metadata[str(source_path)] = {
                'signature': current_signature,
                'original_shape': list(image.shape),
                'patch_paths': patch_paths,
                'hw_size': self.hw_size,
                'depth_size': self.depth_size,
            }
            self.original_shapes.append(image.shape)
            self.patch_paths.extend(patch_paths)
        else:
            # Load existing patch paths from metadata
            patch_paths = self.source_metadata[str(source_path)]['patch_paths']
            shape = self.source_metadata[str(source_path)]['original_shape']
            self.patch_paths.extend(patch_paths)
            self.original_shapes.append(shape)

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


