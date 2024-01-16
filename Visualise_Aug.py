import math

import matplotlib.pyplot as plt
from Components import DataComponents

# The dataset folder you are going to exam the data augmentation with.
INPUT = 'Datasets/train'
# The csv file containing the parameters for image augmentation.
CSV = "Augmentation Parameters.csv"


if __name__ == "__main__":
    dataset = DataComponents.TrainDatasetInstance(INPUT, CSV, 1)
    num_data = len(dataset.img_tensors)
    num_copies = 12
    for i in range(0, num_data):
        images = []
        labels = []
        # 1600 x 900
        plt.figure(figsize=(16,9))
        image_name = dataset.file_list[i][0]
        plt.suptitle(f'{image_name}')
        rows = math.floor(math.sqrt(num_copies*2))
        cols = math.ceil(num_copies/rows)
        for k in range(0, num_copies):
            pair = dataset.__getitem__(i)
            image = pair[0][:, 0:1, :, :].squeeze()
            label = pair[2][:, 0:1, :, :].squeeze()

            # Plot Image
            plt.subplot(rows, 2 * cols, 2 * k + 1)
            plt.imshow(image.cpu().numpy(), cmap='gist_gray')
            plt.axis('off')

            # Plot Label
            plt.subplot(rows, 2 * cols, 2 * k + 2)
            plt.imshow(label.cpu().numpy(), cmap='gist_gray')
            #plt.colorbar()
            plt.axis('off')

        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce spacing between subplots
        plt.show()