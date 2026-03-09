import torch
import pandas as pd

from Components import Augmentations, DataComponents, Metrics
from pathlib import Path

# DataComponents

test_image = torch.from_numpy(DataComponents.path_to_array("Datasets/mid_visualiser/image.tif")[None, :])
test_augmentation_params = pd.read_csv("Augmentation Parameters Anisotropic.csv")

def test_path_to_array():
    image = DataComponents.path_to_array("Datasets/mid_visualiser/image.tif")
    assert image.shape == (40, 144, 144)

def test_path_to_array_nonorm():
    image = DataComponents.path_to_array_nonorm("Datasets/mid_visualiser/image.tif")
    assert image[0].shape == (40, 144, 144)

def test_apply_aug():
    lab_tensor = torch.randint(0, 2, (1, 40, 144, 144), dtype=torch.int32)
    img_tensor = test_image
    augmented_result = DataComponents.apply_aug(img_tensor, lab_tensor, None, test_augmentation_params,
                                                64, 16, 0.001, 0.001)
    assert augmented_result[0].shape == (1, 16, 64, 64)

def test_apply_aug_unsupervised():
    img_tensor = test_image
    augmented_result = DataComponents.apply_aug_unsupervised(img_tensor, test_augmentation_params,
                                                             64, 16)
    assert augmented_result.shape == (1, 16, 64, 64)

def test_make_dataset_tv():
    image_label_pair = DataComponents.make_dataset_tv('Datasets/train')
    assert isinstance(image_label_pair[0][0], Path)

def test_make_dataset_predict():
    image_label_pair = DataComponents.make_dataset_predict('Datasets/predict')
    assert isinstance(image_label_pair[0], Path)

def test_TrainDataset():
    dataset = DataComponents.TrainDataset('Datasets/train', "Augmentation Parameters Anisotropic.csv",
                                          1, 64, 16)
    example_output = dataset.__getitem__((0, None))[0]
    assert example_output.shape == (1, 16, 64, 64)

def test_TrainDatasetChunked():
    dataset = DataComponents.TrainDatasetChunked('Datasets/train', "Augmentation Parameters Anisotropic.csv",
                                                 1, 64, 16)
    example_output = dataset.__getitem__((0, None))[0]
    assert example_output.shape == (1, 16, 64, 64)

def test_UnsupervisedDataset():
    # Use the included predict dataset as data source since the "unsupervised_train" folder is empty by default.
    dataset = DataComponents.UnsupervisedDataset('Datasets/predict', "Augmentation Parameters Anisotropic.csv",
                                                 1, 64, 16)
    example_output = dataset.__getitem__(0)[0]
    assert example_output.shape == (1, 16, 64, 64)

def test_UnsupervisedDatasetChunked():
    dataset = DataComponents.UnsupervisedDatasetChunked('Datasets/predict', "Augmentation Parameters Anisotropic.csv",
                                                 1, 64, 16)
    example_output = dataset.__getitem__(0)[0]
    assert example_output.shape == (1, 16, 64, 64)

def test_ValDataset():
    dataset = DataComponents.ValDataset('Datasets/val', 64, 16, False)
    example_output = dataset.__getitem__(0)[0]
    assert example_output.shape == (1, 16, 64, 64)

def test_ValDatasetChunked():
    dataset = DataComponents.ValDatasetChunked('Datasets/val', 64, 16, False)
    example_output = dataset.__getitem__(0)[0]
    assert example_output.shape == (1, 16, 64, 64)

def test_PredictDataset():
    dataset = DataComponents.PredictDataset('Datasets/predict', 56, 12, 4, 2)
    example_output = dataset.__getitem__(0)
    assert example_output.shape == (1, 16, 64, 64)

def test_PredictDatasetChunked():
    dataset = DataComponents.PredictDatasetChunked('Datasets/predict', 56, 12, 4, 2)
    example_output = dataset.__getitem__(0)
    assert example_output.shape == (1, 16, 64, 64)

# Augmentations

def test_sim_low_res():
    output = Augmentations.sim_low_res(test_image, 2)
    assert output.shape == (1, 40, 144, 144)

def test_gaussian_blur_3d():
    output = Augmentations.gaussian_blur_3d(test_image, kernel_size=3, sigma=0.8)
    assert output.shape == (1, 40, 144, 144)

def test_custom_rand_crop_rotate():
    lab_tensor = torch.randint(0, 2, (1, 40, 144, 144), dtype=torch.int32)
    output = Augmentations.custom_rand_crop_rotate([test_image, lab_tensor], 16, 64, 64)
    assert output[0].shape == (1, 16, 64, 64)

def test_random_gradient():
    output = Augmentations.random_gradient(test_image)
    assert output.shape == (1, 40, 144, 144)

def test_instance_contour_transform():
    test_lab, _, _ = DataComponents.path_to_array_nonorm("Datasets/val/Labels_image.tif")
    output = Augmentations.instance_contour_transform(test_lab)
    assert output.shape == test_lab.shape

def test_nearest_interpolate():
    test_tensor = torch.randint(0, 65536, (1, 1, 16, 64, 64), dtype=torch.uint16)
    output = Augmentations.nearest_interpolate(test_tensor, (32, 128, 128))
    assert (output.shape == (1, 1, 32, 128, 128) and output.dtype == torch.uint16)

def test_edge_replicate_pad():
    output = Augmentations.edge_replicate_pad((test_image,))
    assert output[0].shape == (1, 40, 144, 144)

def test_gaussian_noise():
    output = Augmentations.gaussian_noise(test_image)
    assert output.shape == (1, 40, 144, 144)

# Metrics

def test_BinaryMetrics():
    metric = Metrics.BinaryMetrics("dice+bce")
    test_predicted = torch.randn(1, 1, 16, 64, 64)
    test_gt = torch.randint(0, 2, (1, 1, 16, 64, 64), dtype=torch.float32)
    out = metric(test_predicted, test_gt)
    assert len(out) == 7

def test_instance_segmentation_metrics():
    test_predicted = torch.randint(0, 5, (1, 1, 16, 64, 64))
    test_gt = torch.randint(0, 5, (1, 1, 16, 64, 64))
    out = Metrics.instance_segmentation_metrics(test_predicted, test_gt, 0.5)
    assert len(out) == 5

# MorphologicalFunctions are tested via using the instance_segmentation_simple function since it covers all of them,
# and it's not easy to write on that test them individually

def test_instance_segmentation_simple():
    test_semantic_map = DataComponents.path_to_array("resources_for_pytest/Test_Pixels_Mask.tif")
    test_semantic_map = torch.from_numpy(test_semantic_map)
    test_contour_map = DataComponents.path_to_array("resources_for_pytest/Test_Contour_Mask.tif")
    test_contour_map = torch.from_numpy(test_contour_map)
    segmentation = DataComponents.instance_segmentation_simple(test_semantic_map, test_contour_map)
    assert segmentation.max() == 92

# Welford already covered by test_path_to_array.