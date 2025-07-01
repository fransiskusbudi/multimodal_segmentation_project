import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_loss(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    return 1 - dice

def combined_loss(pred, target):
    """
    Combined loss function for multi-class segmentation
    pred: model output (B, C, H, W, D) where C is number of classes
    target: ground truth (B, 1, H, W, D) with class indices
    """
    # Remove the channel dimension from target
    target = target.squeeze(1)  # Now shape is (B, H, W, D)
    
    # Cross Entropy Loss for multi-class
    ce_loss = nn.CrossEntropyLoss()(pred, target)
    
    # Dice Loss
    pred_softmax = F.softmax(pred, dim=1)
    dice_loss = 0
    for class_idx in range(1, pred_softmax.size(1)):  # Skip background
        pred_mask = pred_softmax[:, class_idx]
        target_mask = (target == class_idx).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        dice_loss += 1 - (2. * intersection + 1e-5) / (union + 1e-5)
    
    dice_loss = dice_loss / (pred_softmax.size(1) - 1)  # Average over classes (excluding background)
    
    return ce_loss + dice_loss

def dice_score(pred, target, epsilon=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1,2,3,4))
    union = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

def iou_score(pred, target, epsilon=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1,2,3,4))
    union = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4)) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean().item()

def accuracy_score(pred, target):
    pred = (pred > 0.5).float()
    target = target.float()
    correct = (pred == target).float().sum()
    total = target.numel()
    return (correct / total).item()

def calculate_iou(pred, target):
    """
    Calculate IoU for multi-class segmentation
    pred: model output (B, C, H, W, D)
    target: ground truth (B, 1, H, W, D)
    """
    # Remove the channel dimension from target
    target = target.squeeze(1)  # Now shape is (B, H, W, D)
    
    pred = torch.argmax(pred, dim=1)  # Convert to class indices
    iou = 0
    valid_classes = 0
    
    for class_idx in range(1, pred.size(1)):  # Skip background
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        # Only include classes that are present in the target
        if target_mask.sum() > 0:
            intersection = (pred_mask & target_mask).sum().float()
            union = pred_mask.sum() + target_mask.sum() - intersection
            iou += (intersection + 1e-5) / (union + 1e-5)
            valid_classes += 1
    
    # Avoid division by zero if no valid classes
    return iou / max(valid_classes, 1)

def calculate_dice(pred, target):
    """
    Calculate Dice coefficient for multi-class segmentation
    pred: model output (B, C, H, W, D)
    target: ground truth (B, 1, H, W, D)
    """
    # Remove the channel dimension from target
    target = target.squeeze(1)  # Now shape is (B, H, W, D)
    
    pred = torch.argmax(pred, dim=1)  # Convert to class indices
    dice = 0
    valid_classes = 0
    
    for class_idx in range(1, pred.size(1)):  # Skip background
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        # Only include classes that are present in the target
        if target_mask.sum() > 0:
            intersection = (pred_mask & target_mask).sum().float()
            union = pred_mask.sum() + target_mask.sum()
            dice += (2. * intersection + 1e-5) / (union + 1e-5)
            valid_classes += 1
    
    # Avoid division by zero if no valid classes
    return dice / max(valid_classes, 1)

def calculate_accuracy(pred, target):
    """
    Calculate accuracy for multi-class segmentation
    pred: model output (B, C, H, W, D)
    target: ground truth (B, 1, H, W, D)
    """
    # Remove the channel dimension from target
    target = target.squeeze(1)  # Now shape is (B, H, W, D)
    
    pred = torch.argmax(pred, dim=1)  # Convert to class indices
    return (pred == target).float().mean()

def calculate_metrics(pred, target):
    dice = dice_score(pred, target)
    iou = iou_score(pred, target)
    acc = accuracy_score(pred, target)
    return dice, iou, acc

def tversky_loss(pred, target, alpha=0.5, beta=0.5, epsilon=1e-6):
    """
    Tversky loss for multi-class segmentation
    pred: model output (B, C, H, W, D) where C is number of classes
    target: ground truth (B, 1, H, W, D) with class indices
    alpha, beta: Tversky parameters (default 0.5, 0.5 for Dice)
    """
    target = target.squeeze(1)  # (B, H, W, D)
    pred_softmax = F.softmax(pred, dim=1)
    tversky_loss = 0
    for class_idx in range(1, pred_softmax.size(1)):  # Skip background
        pred_mask = pred_softmax[:, class_idx]
        target_mask = (target == class_idx).float()
        tp = (pred_mask * target_mask).sum()
        fp = (pred_mask * (1 - target_mask)).sum()
        fn = ((1 - pred_mask) * target_mask).sum()
        tversky = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
        tversky_loss += 1 - tversky
    tversky_loss = tversky_loss / (pred_softmax.size(1) - 1)
    return tversky_loss

def combined_ce_tversky_loss(pred, target, alpha=0.7, beta=0.3):
    """
    Combined CrossEntropy and Tversky loss for multi-class segmentation
    pred: model output (B, C, H, W, D) where C is number of classes
    target: ground truth (B, 1, H, W, D) with class indices
    """
    target_ = target.squeeze(1)
    ce_loss = nn.CrossEntropyLoss()(pred, target_)
    tversky = tversky_loss(pred, target, alpha=alpha, beta=beta)
    return 0.3 * ce_loss + 0.7 * tversky
 