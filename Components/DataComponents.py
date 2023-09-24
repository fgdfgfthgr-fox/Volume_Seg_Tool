import math
import os
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms.functional as T_F
from . import Augmentations as Aug
import numpy as np
import imageio
import random
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_label_fname(fname):
    return 'Labels_' + fname

# 输入图像或者标签的路径，得到已标准化的图像张量或者标签的张量
def path_to_tensor(path, label=False):
    # imread()将文件读取成一个numpy array
    # ToTensor()对16位图不方便，因此才用这招
    img = imageio.v3.imread(path)
    if not label:
        max_value = np.iinfo(img.dtype).max if img.dtype.kind == 'u' else np.finfo(img.dtype).max
        img = (img/max_value).astype(np.float32)
    # from_numpy()则将numpy array转换成张量
    return torch.from_numpy(img)


def apply_aug(img_tensor, lab_tensor, augmentation_params):
    for _, row in augmentation_params.iterrows():
        k = row['Augmentation']
        if k == 'Image Depth':
            depth = int(row['Value'])
        elif k == 'Image Height':
            height = int(row['Value'])
        elif k == 'Image Width':
            width = int(row['Value'])

    for _, row in augmentation_params.iterrows():
        augmentation_method, prob = row['Augmentation'], row['Probability']
        if augmentation_method == 'Vertical Flip' and random.random() < prob:
            img_tensor, lab_tensor = T_F.vflip(img_tensor), T_F.vflip(lab_tensor)
        elif augmentation_method == 'Horizontal Flip' and random.random() < prob:
            img_tensor, lab_tensor = T_F.hflip(img_tensor), T_F.hflip(lab_tensor)
        elif augmentation_method == 'Rescaling':
            if random.random() > prob:
                scale = 1
            else:
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

    img_tensor = torch.squeeze(img_tensor)
    for _, row in augmentation_params.iterrows():
        augmentation_method, prob = row['Augmentation'], row['Probability']
        if augmentation_method == 'Gaussian Blur' and random.random() < prob:
            img_tensor = Aug.gaussian_blur_3d(img_tensor, int(row['Value']))
        elif augmentation_method == 'Adjust Contrast' and random.random() < prob:
            img_tensor = Aug.adj_contrast_3d(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Adjust Gamma' and random.random() < prob:
            img_tensor = Aug.adj_gamma_3D(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
        elif augmentation_method == 'Simulate Low Resolution' and random.random() < prob:
            img_tensor = Aug.sim_low_res_3D(img_tensor, random.uniform(row['Low Bound'], row['High Bound']))
    img_tensor = img_tensor[None, :]
    return img_tensor, lab_tensor

# 输入图像文件夹和标签文件夹的路径，生成图像跟标签有一一对应关系的文件清单
def make_dataset_tv(image_dir, extensions=IMG_EXTENSIONS):
    image_label_pair = []
    image_files = os.listdir(image_dir)
    for fname in sorted(image_files):
        if has_file_allowed_extension(fname, extensions):
            if not "Labels_" in fname:
                path = os.path.join(image_dir, fname)
                label_path = os.path.join(image_dir, 'Labels_' + fname)
                image_label_pair.append((path, label_path))
    # image_label_pair example：
    # [
    # ('Datasets\\train\\img\\testimg1.tif', 'Datasets\\train\\img\\Labels_testimg1.tif'),
    # ('Datasets\\train\\img\\testimg2.tif', 'Datasets\\train\\img\\Labels_testimg2.tif')
    # ]
    return image_label_pair


# 输入图像路径，生成包含图像路径的文件清单
def make_dataset_predict(image_dir, extensions=IMG_EXTENSIONS):
    path_list = []
    image_dir = os.path.expanduser(image_dir)
    for root, _, fnames in sorted(os.walk(image_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.normpath(os.path.join(root, fname))
                path_list.append(path)
    # pic_list的例子
    # ['Datasets\\predict\\testpic1.tif',
    #  'Datasets\\predict\\testpic2.tif']
    return path_list


# 自定义的数据集结构，用于存储训练数据，由于训练数据大小可能不一，所以也顺便完成batch操作
class Train_Dataset_OUTDATED(torch.utils.data.Dataset):
    def __init__(self, images_dir, batch_size):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        self.img_tensors = []
        self.lab_tensors = []
        for idx in range(self.num_files):
            # Convert the image and label path to tensors
            img_tensor = path_to_tensor(self.file_list[idx][0], label=False)
            lab_tensor = path_to_tensor(self.file_list[idx][1], label=True)
            lab_tensor = lab_tensor.long()
            # Append the tensors to the list
            self.img_tensors.append(img_tensor)
            self.lab_tensors.append(lab_tensor)

        self.unique_shapes = set([item.shape for item in self.img_tensors])
        self.img_batches = self._create_batches(self.img_tensors, batch_size)
        self.lab_batches = self._create_batches(self.lab_tensors, batch_size)

        super().__init__()

    def _create_batches(self, tensors, batch_size):
        batches = []
        for shape in self.unique_shapes:
            # Get tensors with the current shape
            tensors_batch = [t for t in tensors if t.shape == shape]
            num_tensors = len(tensors_batch)
            num_batches = math.ceil(num_tensors / batch_size)
            # Stack the tensors with the current shape
            tensors_batch = torch.stack(tensors_batch, dim=0)
            # Reshape the batch to match the desired batch size
            tensors_batch = tensors_batch.reshape((-1,) + shape)
            # Split the reshaped batch into smaller batches with the desired batch size
            tensors_batch = torch.split(tensors_batch, batch_size, dim=0)
            # Extend the list of batches
            batches.extend(tensors_batch)
        return batches

    def __len__(self):
        return len(self.img_batches)

    def __getitem__(self, idx):
        # Get the image and label tensors at the specified index
        img_tensor = self.img_batches[idx]
        lab_tensor = self.lab_batches[idx]
        img_tensor = img_tensor[:, None, :]
        return img_tensor, lab_tensor


class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, augmentation_csv, train_multiplier=1):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        self.img_tensors = [path_to_tensor(item[0], label=False) for item in self.file_list]
        self.lab_tensors = [path_to_tensor(item[1], label=True) for item in self.file_list]
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


class Val_Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, depth=64, height=128, width=128):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        tensors_pairs = [(path_to_tensor(item[0], label=False), path_to_tensor(item[1], label=True)) for item in self.file_list]
        self.chopped_tensor_pairs = []
        for pairs in tensors_pairs:
            self.depth, self.height, self.width = pairs[0].shape
            self.depth_multiplier = math.ceil(self.depth / depth)
            self.height_multiplier = math.ceil(self.height / height)
            self.width_multiplier = math.ceil(self.width / width)
            self.total_multiplier = self.depth_multiplier * self.height_multiplier * self.width_multiplier
            # Loop through each depth, height, and width index
            for depth_idx in range(self.depth_multiplier):
                for height_idx in range(self.height_multiplier):
                    for width_idx in range(self.width_multiplier):
                        # Calculate the start and end indices for depth, height, and width
                        depth_start = int((depth - ((depth * self.depth_multiplier - self.depth) / (
                                    self.depth_multiplier - 1))) * depth_idx)
                        depth_end = depth_start + depth
                        height_start = math.floor((height - ((height * self.height_multiplier - self.height) / (
                                    self.height_multiplier - 1))) * height_idx)
                        height_end = height_start + height
                        width_start = math.floor((width - ((width * self.width_multiplier - self.width) / (
                                    self.width_multiplier - 1))) * width_idx)
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

    
class Val_Dataset_OUTDATED(torch.utils.data.Dataset):
    file_list: list

    def __init__(self, images_dir, batch_size):
        # Get a list of file paths for images and labels
        self.file_list = make_dataset_tv(images_dir)
        self.num_files = len(self.file_list)
        self.img_tensors = []
        self.lab_tensors = []
        for idx in range(self.num_files):
            # Convert the image and label path to tensors
            img_tensor = path_to_tensor(self.file_list[idx][0], label=False)
            lab_tensor = path_to_tensor(self.file_list[idx][1], label=True)
            lab_tensor = lab_tensor.long()
            # Append the tensors to the list
            self.img_tensors.append(img_tensor)
            self.lab_tensors.append(lab_tensor)

        self.unique_shapes = set([item.shape for item in self.img_tensors])
        self.img_batches = self._create_batches(self.img_tensors, batch_size)
        self.lab_batches = self._create_batches(self.lab_tensors, batch_size)

        super().__init__()

    def _create_batches(self, tensors, batch_size):
        batches = []
        for shape in self.unique_shapes:
            # Get tensors with the current shape
            tensors_batch = [t for t in tensors if t.shape == shape]
            num_tensors = len(tensors_batch)
            num_batches = math.ceil(num_tensors / batch_size)
            # Stack the tensors with the current shape
            tensors_batch = torch.stack(tensors_batch, dim=0)
            # Reshape the batch to match the desired batch size
            tensors_batch = tensors_batch.reshape((-1,) + shape)
            # Split the reshaped batch into smaller batches with the desired batch size
            tensors_batch = torch.split(tensors_batch, batch_size, dim=0)
            # Extend the list of batches
            batches.extend(tensors_batch)
        return batches

    def __len__(self):
        return len(self.img_batches)

    def __getitem__(self, idx):
        img_tensor = self.img_batches[idx]
        lab_tensor = self.lab_batches[idx]
        # 手动给张量加上一个"Channel"维度，以便修复需要Channel的问题
        img_tensor = img_tensor[:, None, :].to(torch.float32)
        return img_tensor, lab_tensor


# 自定义的数据集结构，用于存储预测数据
class Predict_Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, hw_size=512, depth_size=12, hw_overlap=32):
        self.file_list = make_dataset_predict(images_dir)
        self.hw_size = hw_size
        self.depth_size = depth_size
        self.hw_overlap = hw_overlap
        self.img_list = [path_to_tensor(file, label=False) for file in self.file_list]
        # Get the size of all tensors in img_list, assume all tensors in img_list are the same size
        self.depth = self.img_list[0].shape[0]
        self.height = self.img_list[0].shape[1]
        self.width = self.img_list[0].shape[2]
        # Calculate the multipliers for padding and cropping
        self.depth_multiplier = math.ceil(self.depth / self.depth_size)
        self.height_multiplier = math.ceil(self.height / self.hw_size)
        self.width_multiplier = math.ceil(self.width / self.hw_size)
        self.total_multiplier = self.depth_multiplier * self.height_multiplier * self.width_multiplier
        # Padding and cropping
        self.padded_img_list = []
        for img_tensor in self.img_list:
            paddings = (self.hw_overlap, self.width_multiplier * self.hw_size + self.hw_overlap - self.width,
                        self.hw_overlap, self.height_multiplier * self.hw_size + self.hw_overlap - self.height,
                        0, 0)
            img_tensor = img_tensor[None, :]
            img_tensor = F.pad(img_tensor, paddings, mode="constant")
            # Loop through each depth, height, and width index
            for depth_idx in range(self.depth_multiplier):
                for height_idx in range(self.height_multiplier):
                    for width_idx in range(self.width_multiplier):
                        # Calculate the start and end indices for depth, height, and width
                        depth_start = min(depth_idx * self.depth_size, self.depth - self.depth_size)
                        depth_end = min(depth_start + self.depth_size, self.depth)
                        height_start = height_idx * self.hw_size
                        height_end = height_start + self.hw_size + self.hw_overlap * 2
                        width_start = width_idx * self.hw_size
                        width_end = width_start + self.hw_size + self.hw_overlap * 2
                        cropped_tensor = img_tensor[:, depth_start:depth_end, height_start:height_end, width_start:width_end]
                        self.padded_img_list.append(cropped_tensor)
        super().__init__()

    def __len__(self):
        return len(self.file_list) * self.total_multiplier

    def __getitem__(self, idx):
        return self.padded_img_list[idx].to(torch.float32)

    def __getoriginalvol__(self):
        return self.img_list[0].shape


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
        self.depth_multiplier = math.ceil(self.depth / depth)
        self.height_multiplier = math.ceil(self.height / height)
        self.width_multiplier = math.ceil(self.width / width)
        self.total_multiplier = self.depth_multiplier * self.height_multiplier * self.width_multiplier
        self.leave_out_list = []
        # Loop through each depth, height, and width index
        for depth_idx in range(self.depth_multiplier):
            for height_idx in range(self.height_multiplier):
                for width_idx in range(self.width_multiplier):
                    # Calculate the start and end indices for depth, height, and width
                    depth_start = int((depth-((depth*self.depth_multiplier-self.depth)/(self.depth_multiplier-1)))*depth_idx)
                    depth_end = depth_start + depth
                    height_start = math.floor((height-((height*self.height_multiplier-self.height)/(self.height_multiplier-1)))*height_idx)
                    height_end = height_start + height
                    width_start = math.floor((width-((width*self.width_multiplier-self.width)/(self.width_multiplier-1)))*width_idx)
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


def stitch_output_volumes(output_volumes, original_volume, hw_size=512, depth_size=12, overlap_size=32):
    depth = original_volume[0]
    height = original_volume[1]
    width = original_volume[2]
    depth_multiplier = math.ceil(depth / depth_size)
    height_multiplier = math.ceil(height / hw_size)
    width_multiplier = math.ceil(width / hw_size)
    total_multiplier = depth_multiplier * height_multiplier * width_multiplier
    result_volume = torch.zeros((depth_multiplier*depth_size,
                                 height_multiplier*hw_size,
                                 width_multiplier*hw_size), dtype=torch.int8)
    for i in range(total_multiplier):
        tensors_in_1_layer = height_multiplier * width_multiplier
        depth_idx = math.floor(i / tensors_in_1_layer) % depth_multiplier
        height_idx = math.floor(i / width_multiplier) % height_multiplier
        width_idx = i % width_multiplier
        tensor_work_with = output_volumes[i][:,
                                             overlap_size:-overlap_size,
                                             overlap_size:-overlap_size]
        depth_start = min(depth_idx * depth_size, depth - depth_size)
        depth_end = min(depth_start + depth_size, depth)
        height_start = height_idx * hw_size
        height_end = height_start + hw_size
        width_start = width_idx * hw_size
        width_end = width_start + hw_size
        result_volume[depth_start:depth_end, height_start:height_end, width_start:width_end] = tensor_work_with
    result_volume = result_volume[0:depth, 0:height, 0:width]
    return result_volume


def predictions_to_final_img(predictions, direc, original_volume, hw_size=512, depth_size=12, hw_overlap=32):
    tensor_list = []
    for prediction in predictions:  # 将这些输出张量项目从predictions里拿出来
        # 分割prediction，形成包含了多个元素（图片张量）的元组，每个元素的Batch维度都是1
        splitted = torch.split(prediction, split_size_or_sections=1, dim=0)
        for single_tensor in splitted:  # 将这个元组里每个单独元素（图片张量）拆分出来
            # It appears that if the final output is a volume, I will need to squeeze the first dimension(Batch)
            single_tensor = torch.squeeze(single_tensor, (0, 1))
            list.append(tensor_list, single_tensor)

    full_image = stitch_output_volumes(tensor_list, original_volume, hw_size, depth_size, hw_overlap)
    array = np.asarray(full_image)
    imageio.v3.imwrite(uri=f'{direc}/full_prediction.tif', image=np.uint8(array))


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
