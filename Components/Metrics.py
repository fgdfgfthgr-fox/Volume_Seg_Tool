import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import find_objects
from numba import njit, prange




class BinaryMetrics(nn.Module):
    def __init__(self, loss_mode: str, smooth=1024.0):
        """
        Initializes the BinaryMetrics module. Which can be set for using dice loss (for semantic map)
        or focal loss (for contour map).

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
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        valid = targets != 0
        inputs = inputs[valid]
        targets = targets[valid]  # now only 1 or 2

        # In-place remap: 1 -> 1, 2 -> 0
        targets.sub_(1)  # fg 0, bg 1
        targets.mul_(-1).add_(1)  # fg 1, bg 0

        return inputs, targets

    def calculate_iou_loss(self, probs, targets, hard_pred):
        # All inputs are flat 1-D tensors
        # Soft intersection/union for differentiable Dice loss
        intersection_s = 2 * torch.dot(targets, probs) + self.smooth
        union_s = targets.sum() + probs.sum() + self.smooth
        loss = 1 - (intersection_s / union_s)

        # Hard intersection/union for logging
        intersection = 2 * targets[hard_pred].sum()
        union = hard_pred.sum() + targets.sum()

        return intersection, union, loss


    def calculate_other_metrices(self, probs, targets, hard_pred):
        ecs = self.expected_calibration_error(probs, targets)

        # Compute TP/FN/FP/TN from aggregate sums to avoid
        # multiple full-size product tensors.
        tp = targets[hard_pred].sum()
        fn = targets.sum() - tp
        fp = hard_pred.sum() - tp
        tn = targets.numel() - targets.sum() - hard_pred.sum() + tp

        return tp.detach(), fn.detach(), tn.detach(), fp.detach(), ecs

    @staticmethod
    def expected_calibration_error(pred, target, n_bins=10):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        total = pred.numel()
        ece = 0.0

        for i in range(n_bins):
            lower = i / n_bins
            upper = (i + 1) / n_bins

            if i == n_bins - 1:
                mask = (pred >= lower) & (pred <= upper)
            else:
                mask = (pred >= lower) & (pred < upper)

            n_bin = mask.sum().item()
            if n_bin == 0:
                continue

            avg_confidence = pred[mask].mean().item()
            avg_accuracy = target[mask].mean().item()

            ece += (n_bin / total) * abs(avg_confidence - avg_accuracy)

        return ece


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
        else:
            predict = predict.reshape(-1)
            target = target.reshape(-1)
        # Placeholder for modes where metrics are not computed
        nan = torch.tensor(float("nan"), device=predict.device, dtype=predict.dtype)
        if self.loss_mode == "bce_no_dice":
            # Scale down to 20% since it's used for unsupervised learning and is often much higher than supervised
            bce_loss = F.binary_cross_entropy_with_logits(predict, target, reduction='mean') * 0.2
            return bce_loss, nan, nan, nan, nan, nan, nan, nan

        if self.loss_mode == "focal":
            bce = F.binary_cross_entropy_with_logits(predict, target, reduction="none")
            focal_weight = (1 - torch.exp(-bce)) ** 1.333
            loss = (focal_weight * bce).mean()
            probs = torch.sigmoid(predict)
            hard_pred = probs >= 0.5
            with torch.no_grad():
                intersection, union, _ = self.calculate_iou_loss(probs, target, hard_pred)
                tp, fn, tn, fp, ecs = self.calculate_other_metrices(probs, target, hard_pred)
        elif self.loss_mode == "dice":
            probs = torch.sigmoid(predict)
            hard_pred = probs >= 0.5
            intersection, union, loss = self.calculate_iou_loss(predict, target, hard_pred)
            with torch.no_grad():
                tp, fn, tn, fp, ecs = self.calculate_other_metrices(predict, target, hard_pred)
        elif self.loss_mode == "dice+bce":
            bce_loss = F.binary_cross_entropy_with_logits(predict, target, reduction="mean")
            probs = torch.sigmoid(predict)
            hard_pred = probs >= 0.5
            intersection, union, dice_loss = self.calculate_iou_loss(probs, target, hard_pred)
            loss = 0.1 * dice_loss + 1.9 * bce_loss
            with torch.no_grad():
                tp, fn, tn, fp, ecs = self.calculate_other_metrices(predict, target, hard_pred)
        else:
            raise ValueError("Invalid loss. Use 'focal' or 'dice' or 'dice+bce' or 'bce_no_dice'.")
        return loss, intersection, union, tp, fn, tn, fp, ecs



@njit(parallel=True, fastmath=True)
def compute_iou_matrix(gt_bboxes, pred_bboxes, gt_volumes, pred_volumes,
                       gt_labels, pred_labels, gt_map, pred_map, iou_matrix):
    """
    Fill iou_matrix with IoU values for all GT-pred pairs whose bounding boxes overlap.
    Non-overlapping pairs get IoU = 0.
    """
    n_gt = gt_bboxes.shape[0]
    n_pred = pred_bboxes.shape[0]

    for i in prange(n_gt):
        for j in range(n_pred):
            # Quick bounding box overlap check
            if (pred_bboxes[j, 3] < gt_bboxes[i, 0] or pred_bboxes[j, 0] > gt_bboxes[i, 3] or
                pred_bboxes[j, 4] < gt_bboxes[i, 1] or pred_bboxes[j, 1] > gt_bboxes[i, 4] or
                pred_bboxes[j, 5] < gt_bboxes[i, 2] or pred_bboxes[j, 2] > gt_bboxes[i, 5]):
                iou_matrix[i, j] = 0.0
                continue

            # Compute intersection over the overlapping region
            z_min = max(gt_bboxes[i, 0], pred_bboxes[j, 0])
            z_max = min(gt_bboxes[i, 3], pred_bboxes[j, 3])
            y_min = max(gt_bboxes[i, 1], pred_bboxes[j, 1])
            y_max = min(gt_bboxes[i, 4], pred_bboxes[j, 4])
            x_min = max(gt_bboxes[i, 2], pred_bboxes[j, 2])
            x_max = min(gt_bboxes[i, 5], pred_bboxes[j, 5])

            intersection = 0
            for z in range(z_min, z_max + 1):
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        if gt_map[z, y, x] == gt_labels[i] and pred_map[z, y, x] == pred_labels[j]:
                            intersection += 1

            union = gt_volumes[i] + pred_volumes[j] - intersection
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0


def instance_segmentation_metrics(pred_map, gt_map, iou_threshold):
    """
    Simple metrics for evaluating instance segmentation. Based on the following principles:
    An instance in the result is considered as a TP if it overlaps with an instance in the ground truth and if this overlapping,
    which is measured by an IOU metric voxel-wise, is higher than a selected threshold.
    If we have multiple instances for one ground truth object,
    the one with the highest IOU is considered as the TP and all the others are counted as FP.

    Args:
        pred_map (np.ndarray): Predicted segmentation map, shape (D, H, W), integer labels.
        gt_map (np.ndarray): Ground truth map, same shape, integer labels.
        iou_threshold (float): IoU threshold for a prediction to be considered a true positive.

    Returns:
        tuple: (tpr, fpr, fnr, precision, recall) as Python floats.
    """
    assert pred_map.shape == gt_map.shape, "Prediction and ground truth maps size mismatch!"

    # Get unique object labels (0 is background)
    gt_labels = np.unique(gt_map)
    gt_labels = gt_labels[gt_labels != 0]
    pred_labels = np.unique(pred_map)
    pred_labels = pred_labels[pred_labels != 0]

    n_gt = len(gt_labels)
    n_pred = len(pred_labels)

    # If no ground truth objects, all metrics are zero
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # If no predicted objects, every GT is a false negative
    if n_pred == 0:
        return 0.0, 0.0, 1.0, 0.0, 0.0

    # Get bounding box slices for each object
    gt_slices = find_objects(gt_map)
    pred_slices = find_objects(pred_map)

    # Build arrays: bounding boxes (min/max inclusive), volumes, labels
    gt_bboxes = np.zeros((n_gt, 6), dtype=np.int64)
    gt_volumes = np.zeros(n_gt, dtype=np.int64)
    for idx, label in enumerate(gt_labels):
        sl = gt_slices[label - 1]
        # sl is a tuple of slices (z, y, x)
        gt_bboxes[idx, 0] = sl[0].start  # min depth
        gt_bboxes[idx, 1] = sl[1].start  # min height
        gt_bboxes[idx, 2] = sl[2].start  # min width
        gt_bboxes[idx, 3] = sl[0].stop - 1  # max depth (inclusive)
        gt_bboxes[idx, 4] = sl[1].stop - 1  # max height
        gt_bboxes[idx, 5] = sl[2].stop - 1  # max width
        gt_volumes[idx] = np.sum(gt_map[sl] == label)

    pred_bboxes = np.zeros((n_pred, 6), dtype=np.int64)
    pred_volumes = np.zeros(n_pred, dtype=np.int64)
    for idx, label in enumerate(pred_labels):
        sl = pred_slices[label - 1]
        pred_bboxes[idx, 0] = sl[0].start
        pred_bboxes[idx, 1] = sl[1].start
        pred_bboxes[idx, 2] = sl[2].start
        pred_bboxes[idx, 3] = sl[0].stop - 1
        pred_bboxes[idx, 4] = sl[1].stop - 1
        pred_bboxes[idx, 5] = sl[2].stop - 1
        pred_volumes[idx] = np.sum(pred_map[sl] == label)

    # Compute pairwise IoU matrix (parallelized with Numba)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    compute_iou_matrix(gt_bboxes, pred_bboxes, gt_volumes, pred_volumes,
                       gt_labels, pred_labels, gt_map, pred_map, iou_matrix)

    # Sequential greedy matching (same logic as original PyTorch version)
    tp = 0
    fp = 0
    matched_pred_to_gt = {}  # pred_idx -> (iou, gt_idx)

    for i in range(n_gt):
        best_iou = 0.0
        best_pred_idx = -1

        for j in range(n_pred):
            iou = iou_matrix[i, j]
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j

        if best_iou > iou_threshold:
            if best_pred_idx not in matched_pred_to_gt:
                tp += 1
                matched_pred_to_gt[best_pred_idx] = (best_iou, i)
            else:
                fp += 1
                stored_iou, _ = matched_pred_to_gt[best_pred_idx]
                if best_iou > stored_iou:
                    matched_pred_to_gt[best_pred_idx] = (best_iou, i)
        else:
            if best_pred_idx != -1:  # there was at least some positive overlap
                fp += 1

    fn = n_gt - tp

    # Compute final metrics
    tpr = tp / n_gt if n_gt > 0 else 0.0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    fnr = fn / n_gt if n_gt > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return tpr, fpr, fnr, precision, recall
