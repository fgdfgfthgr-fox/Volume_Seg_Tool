import math

import numpy as np
import torch
from skimage.io import imread_collection, imsave
from joblib import Parallel, delayed
from tqdm import tqdm
import imagej
import torch.nn as nn
import torch.nn.functional as F
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

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    img_gt: segmentation, shape = (batch_size, x, y, z)
    out_shape: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    from scipy.ndimage import distance_transform_edt
    from skimage import segmentation as skimage_seg

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]): # channel
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf

class BDLoss(nn.Module):
    def __init__(self, **_):
        """
        compute boundary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, inputs, labels, **_):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        with torch.no_grad():
            if len(inputs.shape) != len(labels.shape):
                labels = labels.view((labels.shape[0], 1, *labels.shape[1:]))

            if all([i == j for i, j in zip(inputs.shape, labels.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = labels
            else:
                labels = labels.long()
                y_onehot = torch.zeros(inputs.shape)
                if inputs.device.type == "cuda":
                    y_onehot = y_onehot.cuda(inputs.device.index)
                y_onehot.scatter_(1, labels, 1)
            gt_sdf = compute_sdf(y_onehot.cpu().numpy(), inputs.shape)

        phi = torch.from_numpy(gt_sdf)
        if phi.device != inputs.device:
            phi = phi.to(inputs.device).type(torch.float32)
        # pred = net_output[:, 1:, ...].type(torch.float32)
        # phi = phi[:,1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", inputs[:, 1:, ...], phi[:, 1:, ...])
        bd_loss = multipled.mean()

        return bd_loss


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


class BinaryMetricsForList(nn.Module):
    def __init__(self, loss_mode: str, smooth=1024):
        """
        A loss module designed to calculate evaluation related metrics as well as loss. Dealing with a list of predicted and a single target.

        Args:
            loss_mode (str): A string indicating whether to use focal loss ("focal") or dice+bce ("dice+bce").
            smooth (float): A smoothing factor for numerical stability (default is 1024, very large, explained in the code)
        """
        super(BinaryMetricsForList, self).__init__()
        self.loss_mode = loss_mode
        self.smooth = smooth

    @staticmethod
    def sparse_preprocessing(predicts, target):
        # In sparse label cases:
        # Predict: 1 = Foreground, 0 = Background. Can be any number in between.
        # Target: 0 = Unlabelled, 1 = Foreground, 2 = Background
        target = target.flatten()
        inputs = [predict.flatten() for predict in predicts]
        target = torch.where(target == 0, math.nan, target)
        target = 1 - (target - 1)
        known_label = ~torch.isnan(target)
        inputs = [input[known_label] for input in inputs]
        targets = target[known_label]
        return inputs, targets

    def calculate_iou_loss(self, predict: torch.Tensor, target: torch.Tensor):
        intersection_s = 2 * torch.sum(target * predict) + self.smooth
        # Huge smooth to prevent loss jump when the ground truth foreground is very low or 0
        union_s = torch.sum(predict) + torch.sum(target) + self.smooth
        loss = 1 - (intersection_s / union_s)
        with torch.no_grad():
            predict_map = torch.where(predict >= 0.5, 1, 0).to(torch.int8)
            intersection = 2 * torch.sum(target * predict_map)
            union = torch.sum(predict_map) + torch.sum(target)
        return intersection, union, loss

    @staticmethod
    def calculate_other_metrices(inputs: torch.Tensor, targets: torch.Tensor):
        inputs = torch.where(inputs >= 0.5, 1, 0).to(torch.int8)
        true_positives = (inputs*targets).sum().detach()
        false_negatives = ((1-inputs)*targets).sum().detach()
        true_negatives = ((1-inputs)*(1-targets)).sum().detach()
        false_positives = (inputs*(1-targets)).sum().detach()
        return true_positives, false_negatives, true_negatives, false_positives

    def forward(self, predicts: list[torch.Tensor], target: torch.Tensor, weights: list, sparse_label=False):
        """
        Calculate binary classification metrics and loss based on the provided inputs and targets.

        Args:
            predicts (List of torch.Tensor): The predicted binary classification values (B, 1, D, H, W).
            target (torch.Tensor): The target labels (B, 1, D, H, W).
                When `sparse_label` is True: 0 for unlabeled, 1 for foreground, 2 for background
                When `sparse_label` is False: 0.0 for background, 1.0 for foreground
            weights (list): The weight for each input in the list. Should sum up to 1.
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
            predicts, target = self.sparse_preprocessing(predicts, target)

        if self.loss_mode == "focal":
            bce_losses = [F.binary_cross_entropy_with_logits(predict, target, reduction='none') for predict in predicts]
            pts = [torch.exp(-bce_loss) for bce_loss in bce_losses]
            f_losses = [((1-pt) ** 1.333 * bce_loss).mean() for pt, bce_loss in zip(pts, bce_losses)]
            f_loss = sum([loss * weights[i] for i, loss in enumerate(f_losses)])

            with torch.no_grad():
                iou_outs = [self.calculate_iou_loss(F.sigmoid(predict), target) for predict in predicts]
                other_metrics = [self.calculate_other_metrices(F.sigmoid(predict), target) for predict in predicts]
                intersection, union = sum([scale[0] for scale in iou_outs]), sum([scale[1] for scale in iou_outs])
                tp, fn, tn, fp = [sum([scale[i] for scale in other_metrics]) for i in range(4)]
            return f_loss, intersection, union, tp, fn, tn, fp
        elif self.loss_mode == "dice+bce":
            bce_losses = [F.binary_cross_entropy_with_logits(predict, target, reduction='none').mean() for predict in predicts]
            bce_loss = sum([loss * weights[i] for i, loss in enumerate(bce_losses)])
            iou_outs = [self.calculate_iou_loss(F.sigmoid(predict), target) for predict in predicts]
            for i, iou_out in enumerate(iou_outs):
                iou_outs[i][0] *= weights[i]
                iou_outs[i][1] *= weights[i]
                iou_outs[i][2] *= weights[i]
            intersection, union, dice_loss = [sum([scale[i] for scale in iou_outs]) for i in range(3)]
            with torch.no_grad():
                other_metrics = [self.calculate_other_metrices(F.sigmoid(predict), target) for predict in predicts]
                tp, fn, tn, fp = [sum([scale[i] for scale in other_metrics]) for i in range(4)]
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
