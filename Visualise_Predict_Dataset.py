
from Components import DataComponents
import math
import matplotlib.pyplot as plt
import imageio
import numpy as np

# The path to where the image for prediction are.
PATH_TO_PREDICT_DATASET = 'Datasets/predict'
# The height and width of sub-images the prediction image will be cut to. Larger means more accurate segmentation, but takes more VRAM.
HW_SIZE = 128
# Same as above but is depth.
DEPTH_SIZE = 64
# The overlaps in height and width between each adjacent sub-pictures. Larger means more accurate segmentation, but takes more VRAM.
HW_OVERLAP = 16
# Same as above but is depth.
DEPTH_OVERLAP = 8
# Save the generated volume visualization results.
SAVE_VISUALIZATION_RESULTS = True
# Folder which the results will be saved to.
RESULT_FOLDER = 'Visualizations'

if __name__ == "__main__":
    dataset = DataComponents.Predict_Dataset(PATH_TO_PREDICT_DATASET,
                                             hw_size=HW_SIZE, depth_size=DEPTH_SIZE,
                                             hw_overlap=HW_OVERLAP, depth_overlap=DEPTH_OVERLAP)
    num_data = dataset.total_multiplier
    for i in range(0, len(dataset.file_list)):
        images = []
        # 1600 x 900
        plt.figure(figsize=(16, 9))
        image_name = dataset.file_list[i]
        plt.suptitle(f'{image_name}')
        rows = math.floor(math.sqrt(num_data))
        cols = math.ceil(num_data / rows)
        for k in range(0, num_data):
            image = dataset.__getitem__(k)
            image = image.squeeze()
            if SAVE_VISUALIZATION_RESULTS:
                array = np.asarray(image)
                imageio.v3.imwrite(uri=f'{RESULT_FOLDER}/{k}.tif', image=np.float16(array))
            image = image[0:1, :, :]
            images.append(image)

        for j, copy in enumerate(images):
            copy = copy.squeeze()
            plt.subplot(rows, cols, j + 1)
            plt.imshow(copy.cpu().numpy(), cmap='gist_gray')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce spacing between subplots
        plt.show()