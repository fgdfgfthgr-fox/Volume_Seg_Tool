import math

import matplotlib.pyplot as plt
from Components import DataComponents

# The dataset folder you are going to exam the data augmentation with.
INPUT = 'Datasets/train'
# The csv file containing the parameters for image augmentation.
CSV = "Augmentation Parameters.csv"


if __name__ == "__main__":
    dataset = DataComponents.Train_Dataset(INPUT, CSV)
    num_data = len(dataset.img_tensors)
    num_copies = 12
    for i in range(0, num_data):
        images = []
        # 1600 x 900
        plt.figure(figsize=(16,9))
        image_name = dataset.file_list[i][0]
        plt.suptitle(f'{image_name}')
        rows = math.floor(math.sqrt(num_copies))
        cols = math.ceil(num_copies/rows)
        for k in range(0, num_copies):
            image = dataset.__getitem__(i)
            image = image[0][:, 0:1, :, :].squeeze()
            images.append(image)
        for j, copy in enumerate(images):
            copy = copy.squeeze()
            plt.subplot(rows, cols, j + 1)
            plt.imshow(copy.cpu().numpy(), cmap='gist_gray')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce spacing between subplots
        plt.show()