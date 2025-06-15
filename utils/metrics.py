import torch
import numpy as np

def dice_loss(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    return 1 - dice

def combined_loss(pred, target, alpha=0.5):
    bce = torch.nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1 - alpha) * dice

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

def calculate_metrics(pred, target):
    dice = dice_score(pred, target)
    iou = iou_score(pred, target)
    acc = accuracy_score(pred, target)
    return dice, iou, acc
