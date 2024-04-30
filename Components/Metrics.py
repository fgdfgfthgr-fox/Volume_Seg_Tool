import math

import numpy as np
import torch
from skimage.io import imread_collection, imsave
from joblib import Parallel, delayed
from numba import cuda
from tqdm import tqdm
import imagej
import torch.nn as nn
import time


class DiceLoss(nn.Module):
    def __init__(self, average='micro'):
        """
        Initializes the DiceLoss module.

        Args:
            average (str): Averaging method for computing the Dice loss.
                - 'micro': Calculate the metric globally across all samples and classes.
                - 'macro': Calculate the metric for each class separately and average the metrics across classes
                           with equal weights for each class.
        """
        super(DiceLoss, self).__init__()
        self.average = average

    def forward(self, inputs, targets, smooth=1):
        """
        Compute the Dice loss between the input predictions and target labels.

        Args:
            inputs (torch.Tensor): Predictions tensor, typically with values between 0 and 1.
                                   Shape: (batch_size, num_classes, ...)
            targets (torch.Tensor): Target labels tensor with binary values (0 or 1).
                                    Shape: (batch_size, num_classes, ...)
            smooth (float): Smoothing factor to prevent division by zero in the Dice coefficient.

        Returns:
            torch.Tensor: Dice loss value.
        """
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets)
        intersection = intersection.sum()
        union = inputs.sum() + targets.sum()

        if self.average == 'micro':
            dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)
        elif self.average == 'macro':
            class_dice_coefficients = []
            unique_classes = torch.unique(targets)

            for class_label in unique_classes:
                class_inputs = (inputs * (targets == class_label).float()).sum()
                class_targets = (targets == class_label).sum()
                class_dice_coefficient = (2.0 * class_inputs + smooth) / (class_inputs + class_targets + smooth)
                class_dice_coefficients.append(class_dice_coefficient)

            dice_coefficient = torch.mean(torch.stack(class_dice_coefficients))
        else:
            raise ValueError("Invalid average method. Use 'micro' or 'macro'.")

        dice_loss = 1 - dice_coefficient
        return dice_loss


class BinaryMetrics(nn.Module):
    def __init__(self, use_log_cosh=False, sparse_label=False):
        """
        Initializes the BinaryMetrics module.

        Args:
            use_log_cosh (bool): A flag indicating whether to use the Log-Cosh Dice Loss (default is False).
            sparse_label (bool): A flag indicating whether the target labels are sparse (default is False).
        """
        super(BinaryMetrics, self).__init__()
        self.use_log_cosh = use_log_cosh
        self.sparse_label = sparse_label

    def forward(self, inputs, targets, smooth=1):
        """
        Calculate binary classification metrics and loss based on the provided inputs and targets.

        Args:
            inputs (torch.Tensor): The predicted binary classification values (B, 1, D, H, W).
            targets (torch.Tensor): The target labels (B, 1, D, H, W).
                When `sparse_label` is True: 0 for unlabeled, 1 for foreground, 2 for background
                When `sparse_label` is False: 0.0 for background, 1.0 for foreground
            smooth (float, optional): A small smoothing factor to prevent division by zero (default is 1e-8).

        Returns:
            loss (torch.Tensor): The calculated loss value based on the chosen loss function.
            dice_score (torch.Tensor): The Dice score.
            tpr (torch.Tensor): True Positive Rate (Sensitivity).
            tnr (torch.Tensor): True Negative Rate (Specificity).
        """
        # Calculate True Positives, True Negatives, False Positives, False Negatives
        if self.sparse_label:
            # Input(i): (B, 1, D, H, W), float32
            # 1 = Foreground, 0 = Background. Can be any number in between.
            # Target(t): (B, 1, D, H, W), int8
            # 0 = Unlabelled
            # 1 = Foreground
            # 2 = Background
            targets = torch.where(targets == 0, math.nan, targets)
            targets = 1 - (targets - 1)
            known_label = ~torch.isnan(targets)
            inputs = inputs[known_label]
            targets = targets[known_label]
        # In Non-sparse_label cases:
        # Input(i): (B, 1, D, H, W), float32
        # 1 = Foreground, 0 = Background. Can be any number in between.
        # Target(t): (B, 1, D, H, W), float32
        # 1 = Foreground, 0 = Background. Can be any number in between.

        # Calculate Dice Score
        intersection = 2 * torch.sum(targets * inputs) + smooth
        union = torch.sum(inputs) + torch.sum(targets) + (smooth * 2)
        dice_score = intersection / union

        # Calculate TP, FN, TN, FP
        inputs = torch.where(inputs >= 0.5, 1, 0).to(torch.int8)
        #targets = targets.to(torch.int8)

        true_positives = (inputs*targets).sum().detach()
        false_negatives = ((1-inputs)*targets).sum().detach()
        true_negatives = ((1-inputs)*(1-targets)).sum().detach()
        false_positives = (inputs*(1-targets)).sum().detach()

        # Calculate True Positive Rate (TPR) and True Negative Rate (TNR)
        tpr = (true_positives + smooth) / (true_positives + false_negatives + smooth)
        tnr = (true_negatives + smooth) / (true_negatives + false_positives + smooth)

        if dice_score == 1:
            print('Dice Score Too High (==1)! That is unrealistic in most of cases!')
        elif dice_score <= 0.001:
            print(f'Dice Score Too Low! Current stats: intersection={intersection}, union={union}, tp={true_positives}, fn={false_negatives}, tn={true_negatives}, fp={false_positives}')

        if self.use_log_cosh:
            # Calculate Log-Cosh Dice Loss
            loss = torch.log(torch.cosh(1.0 - dice_score))
        else:
            # Calculate Dice Loss
            loss = 1 - dice_score

        return loss, dice_score, tpr, tnr


def get_bounding_boxes(tensor):
    """
    Returns a dictionary of bounding boxes for each object in the tensor.
    Each box is represented as [min_depth, min_height, min_width, max_depth, max_height, max_width].
    """
    objects = tensor.unique()[1:]  # Exclude background
    boxes = {}
    for obj in objects:
        positions = (tensor == obj).nonzero(as_tuple=False)
        mins = torch.min(positions, dim=0).values
        maxs = torch.max(positions, dim=0).values
        boxes[obj] = torch.cat((mins, maxs), dim=0)
    return boxes


def instance_segmentation_metrics(pred_map, gt_map, iou_threshold):
    """
    Simple metrics for evaluating instance segmentation. Based on the following principles:
    An instance in the result is considered as a TP if it overlaps with an instance in the ground truth and if this overlapping,
    which is measured by an IOU metric voxel-wise, is higher than a selected threshold.
    If we have multiple instances for one ground truth object,
    the one with the highest IOU is considered as the TP and all the others are counted as FP.

    Args:
        pred_map (torch.Tensor): The segmentation map to be evaluated. Should have a shape of (Depth, Height, Width)
        gt_map (torch.Tensor): Ground Truth Map. Should be the same shape as pred_map
        iou_threshold (float): threshold for the IOU to be considered as TP

    Returns:
        tpr, fpr, fnr, precision, recall (float): The calculated metrics.
    """
    assert pred_map.shape == gt_map.shape, "Prediction and ground truth maps size mismatch!"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_map = pred_map.to(device=device)
    gt_map = gt_map.to(device=device)

    def calculate_iou(pred, gt):
        intersection = torch.logical_and(pred, gt).sum()
        union = torch.logical_or(pred, gt).sum()
        return intersection / union if union != 0 else 0.0

    pred_boxes = get_bounding_boxes(pred_map)
    gt_boxes = get_bounding_boxes(gt_map)

    tp = torch.tensor(0, dtype=torch.int32, device=device)
    fp = torch.tensor(0, dtype=torch.int32, device=device)
    fn = torch.tensor(len(gt_boxes), dtype=torch.int32, device=device)

    gt_to_pred_iou = {}

    with tqdm(total=len(gt_boxes), desc="Processing", unit="objects") as pbar:
        for gt_object, gt_box in gt_boxes.items():
            gt_object_mask = (gt_map == gt_object)
            best_iou = 0.0
            best_pred_object = None

            for pred_object, pred_box in pred_boxes.items():
                # Quick bounding box overlap check before expensive IOU calculation
                if not (pred_box[3] < gt_box[0] or pred_box[0] > gt_box[3] or
                        pred_box[4] < gt_box[1] or pred_box[1] > gt_box[4] or
                        pred_box[5] < gt_box[2] or pred_box[2] > gt_box[5]):
                    pred_object_mask = (pred_map == pred_object)
                    iou = calculate_iou(pred_object_mask, gt_object_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_object = pred_object

            if best_iou > iou_threshold:
                tp += 1
                if best_pred_object in gt_to_pred_iou and gt_to_pred_iou[best_pred_object][0] < best_iou:
                    fp += 1  # Previous best is now considered FP
                gt_to_pred_iou[best_pred_object] = (best_iou, gt_object)
            else:
                if best_pred_object is not None:
                    fp += 1

            pbar.update(1)

    fp += len(pred_boxes) - len(gt_to_pred_iou)  # All non-matching predictions are FPs
    fn -= tp  # Adjust FN

    tpr = tp / len(gt_boxes) if gt_boxes else 0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0
    fnr = fn / len(gt_boxes) if gt_boxes else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return tpr.cpu(), fpr.cpu(), fnr.cpu(), precision.cpu(), recall.cpu()


def getvrand(fiji_dic: str, ground_truth, predicted):
    """
    Calculate V-Rand (Foreground-restricted Rand Scoring) score given the predicted map and the ground truth map.
    Essentially The Rand error, for more information please see https://imagej.net/plugins/tws/rand-error.\n
    Best used to evaluate membrane segmentation task, e.g. ISBI-2012 segmentation challenge.\n
    ImageJ (Fiji) must be installed in order to use it.\n
    Note: Calculating V-Rand score is fairly slow.\n
    Note2: Require a very specific data format, where 0 is the edge label and 255 is everything else.

    Args:
        fiji_dic (str): Installation directory of your Fiji.app folder.
        ground_truth (torch.Tensor): Ground Truth Map of shape (D, H, W).
        predicted (torch.Tensor): Predicted Map of shape (D, H, W).

    Returns:
        VRand (float): The calculated V-Rand score which range from 0 to 1. 0 is the worst while 1 is the best.
    """
    ij = imagej.init(fiji_dic)
    Language_extension = "BeanShell"
    Ground_data = np.asarray(ground_truth, dtype='float32')
    if np.max(Ground_data) != 1:
        Ground_data = Ground_data / 255.0
    imsave(f'GroundData.tif', Ground_data)

    Result_data = np.asarray(predicted, dtype='float32')
    if np.max(Result_data) != 1:
        Result_data = Result_data / 255.0
    imsave(f'ResultData.tif', Result_data)

    macroVRand = """
import trainableSegmentation.metrics.*;
#@output String VRand
import ij.IJ;
originalLabels=IJ.openImage("GroundData.tif");
proposedLabels=IJ.openImage("ResultData.tif");
metric = new RandError( originalLabels, proposedLabels );
maxThres = 1.0;
maxScore = metric.getMaximalVRandAfterThinning( 0.0, maxThres, 0.1, true );  
VRand = maxScore;
"""

    VRand = ij.py.run_script(Language_extension, macroVRand).getOutput('VRand')
    return VRand


def getvinfo(fiji_dic: str, ground_truth, predicted):
    """
    Calculate V-Info (Foreground-restricted Information Theoretic Scoring) score given the predicted map and the ground truth map.
    For more information please see https://github.com/akhadangi/Segmentation-Evaluation-after-border-thinning.\n
    Best used to evaluate membrane segmentation task, e.g. ISBI-2012 segmentation challenge.\n
    ImageJ (Fiji) must be installed in order to use it.\n
    Note: Calculating V-Info score is fairly slow.\n
    Note2: Require a very specific data format, where 0 is the edge label and 255 is everything else.

    Args:
        fiji_dic (str): Installation directory of your Fiji.app folder.
        ground_truth (torch.Tensor): Ground Truth Map of shape (D, H, W).
        predicted (torch.Tensor): Predicted Map of shape (D, H, W).

    Returns:
        VRand (float): The calculated V-Info score which range from 0 to 1. 0 is the worst while 1 is the best.
    """
    ij = imagej.init(fiji_dic)
    Language_extension = "BeanShell"
    Ground_data = np.asarray(ground_truth, dtype='float32')
    if np.max(Ground_data) != 1:
        Ground_data = Ground_data / 255.0
    imsave(f'GroundData.tif', Ground_data)

    Result_data = np.asarray(predicted, dtype='float32')
    if np.max(Result_data) != 1:
        Result_data = Result_data / 255.0
    imsave(f'ResultData.tif', Result_data)

    macroVInfo = """
import trainableSegmentation.metrics.*;
#@output String VInfo
import ij.IJ;
originalLabels=IJ.openImage("GroundData.tif");
proposedLabels=IJ.openImage("ResultData.tif");
metric = new VariationOfInformation( originalLabels, proposedLabels );
maxThres =1.0;
maxScore = metric.getMaximalVInfoAfterThinning( 0.0, maxThres, 0.1 );  
VInfo = maxScore;
"""

    VInfo = ij.py.run_script(Language_extension, macroVInfo).getOutput('VInfo')
    return VInfo

# test_ground = torch.as_tensor(imageio.v3.imread("/mnt/7018F20D48B6C548/PycharmProjects/Deeplearning/CV/train-labels.tif"))
# test_predicted = torch.as_tensor(imageio.v3.imread("/mnt/7018F20D48B6C548/PycharmProjects/Deeplearning/CV/train-labels_distorted.tif"))
# vrand = getvinfo('/home/fgdfgfthgr/Programmes/Fiji.app', test_ground, test_predicted)
# print(vrand)
