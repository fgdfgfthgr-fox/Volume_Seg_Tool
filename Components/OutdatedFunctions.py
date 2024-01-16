
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