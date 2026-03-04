import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


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
    def __init__(self, loss_mode: str, smooth=1024):
        """
        Initializes the BinaryMetrics module. Which can be set for using Boundary loss (for semantic map)
        or Focal Loss (for contour map).

        Args:
            loss_mode (str): A string indicating whether to use focal loss ("focal")
                             or dice loss ("dice") or dice+bce ("dice+bce") or bce loss without dice calculation ("bce_no_dice")
            smooth (float): A smoothing factor for numerical stability (default is 1024, very large, explained in the code)
        """
        super(BinaryMetrics, self).__init__()
        self.loss_mode = loss_mode
        self.smooth = smooth

    @staticmethod
    def sparse_preprocessing(inputs: torch.Tensor, targets: torch.Tensor):
        # In sparse label cases:
        # Input: 1 = Foreground, 0 = Background. Can be any number in between.
        # Target: 0 = Unlabelled, 1 = Foreground, 2 = Background
        targets = targets.flatten()
        inputs = inputs.flatten()
        targets = torch.where(targets == 0, math.nan, targets)
        targets = 1 - (targets - 1)
        known_label = ~torch.isnan(targets)
        inputs = inputs[known_label]
        targets = targets[known_label]
        return inputs, targets

    def calculate_iou_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        intersection_s = 2 * torch.sum(targets * inputs) + self.smooth
        # Huge smooth to prevent loss jump when the ground truth foreground is very low or 0
        union_s = torch.sum(inputs) + torch.sum(targets) + self.smooth
        loss = 1 - (intersection_s / union_s)
        inputs = torch.where(inputs >= 0.5, True, False)
        intersection = 2 * torch.sum(targets * inputs)
        union = torch.sum(inputs) + torch.sum(targets)
        return intersection, union, loss

    @staticmethod
    def calculate_other_metrices(inputs: torch.Tensor, targets: torch.Tensor):
        inputs = torch.where(inputs >= 0.5, 1, 0).to(torch.int8)
        true_positives = (inputs*targets).sum().detach()
        false_negatives = ((1-inputs)*targets).sum().detach()
        true_negatives = ((1-inputs)*(1-targets)).sum().detach()
        false_positives = (inputs*(1-targets)).sum().detach()
        return true_positives, false_negatives, true_negatives, false_positives

    def forward(self, predict: torch.Tensor, target: torch.Tensor, sparse_label=False):
        """
        Calculate binary classification metrics and loss based on the provided inputs and targets.

        Args:
            predict (torch.Tensor): The predicted binary classification values (B, 1, D, H, W).
            target (torch.Tensor): The target labels (B, 1, D, H, W).
                When `sparse_label` is True: 0 for unlabeled, 1 for foreground, 2 for background
                When `sparse_label` is False: 0.0 for background, 1.0 for foreground
            sparse_label (bool): A flag indicating whether the target labels are sparse (default is False).
            If true, will force to dice loss.

        Returns:
            loss (torch.Tensor): The calculated loss value based on the chosen loss function.
            intersection (torch.Tensor)
            union (torch.Tensor)
            true_positives (torch.Tensor)
            false_negatives (torch.Tensor)
            true_negatives (torch.Tensor)
            false_positives (torch.Tensor)
        """
        # In Non-sparse label cases:
        # Input: 1 = Foreground, 0 = Background. Can be any number in between.
        # Target: 1 = Foreground, 0 = Background. Can be any number in between.
        if sparse_label:
            predict, target = self.sparse_preprocessing(predict, target)
            #return self.dice_loss(inputs, targets)

        if self.loss_mode == "focal":
            BCE_loss = F.binary_cross_entropy_with_logits(predict, target, reduction='none')
            predict = torch.sigmoid(predict)
            pt = torch.exp(-BCE_loss)
            F_loss = (1-pt) ** 1.333 * BCE_loss
            with torch.no_grad():
                intersection, union, _ = self.calculate_iou_loss(predict, target)
                tp, fn, tn, fp = self.calculate_other_metrices(predict, target)
            return F_loss.mean(), intersection, union, tp, fn, tn, fp
        elif self.loss_mode == "bce_no_dice":
            # Scale down to 20% since it's used for unsupervised learning and is often much higher than supervised
            BCE_loss = F.binary_cross_entropy_with_logits(predict, target, reduction='none') * 0.2
            return BCE_loss.mean(), torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan
        elif self.loss_mode == "dice":
            predict = torch.sigmoid(predict)
            intersection, union, loss = self.calculate_iou_loss(predict, target)
            with torch.no_grad():
                tp, fn, tn, fp = self.calculate_other_metrices(predict, target)
            return loss, intersection, union, tp, fn, tn, fp
        elif self.loss_mode == "dice+bce":
            bce_loss = F.binary_cross_entropy_with_logits(predict, target, reduction='none').mean()
            predict = torch.sigmoid(predict)
            intersection, union, dice_loss = self.calculate_iou_loss(predict, target)
            with torch.no_grad():
                tp, fn, tn, fp = self.calculate_other_metrices(predict, target)
            total_loss = 0.1*dice_loss+1.9*bce_loss
            return total_loss, intersection, union, tp, fn, tn, fp
        else:
            raise ValueError("Invalid loss. Use 'boundary' or 'focal' or 'dice' or 'dice+bce'.")


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
                if best_pred_object not in gt_to_pred_iou:
                    tp += 1
                    gt_to_pred_iou[best_pred_object] = (best_iou, gt_object)
                else:
                    fp += 1  # Either the previous best is now considered FP, or the current one is an FP.
                    if gt_to_pred_iou[best_pred_object][0] < best_iou:
                        gt_to_pred_iou[best_pred_object] = (best_iou, gt_object)
            else:
                if best_pred_object is not None:
                    fp += 1

            pbar.update(1)

    fn -= tp  # Adjust FN

    tpr = tp / len(gt_boxes) if gt_boxes else 0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0
    fnr = fn / len(gt_boxes) if gt_boxes else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    del pred_map, gt_map

    return tpr.cpu(), fpr.cpu(), fnr.cpu(), precision.cpu(), recall.cpu()

