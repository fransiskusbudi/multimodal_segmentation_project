#!/usr/bin/env python3
"""
Quick fixes for training issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def improved_combined_loss(pred, target, alpha=0.5, beta=0.5):
    """
    Improved combined loss with better handling of class imbalance
    pred: model output (B, C, H, W, D) where C is number of classes
    target: ground truth (B, 1, H, W, D) with class indices
    """
    # Remove the channel dimension from target
    target = target.squeeze(1)  # Now shape is (B, H, W, D)
    
    # Focal Loss for better handling of class imbalance
    ce_loss = nn.CrossEntropyLoss(reduction='none')(pred, target)
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** 2 * ce_loss
    focal_loss = focal_loss.mean()
    
    # Dice Loss with better numerical stability
    pred_softmax = F.softmax(pred, dim=1)
    dice_loss = 0
    valid_classes = 0
    
    for class_idx in range(1, pred_softmax.size(1)):  # Skip background
        pred_mask = pred_softmax[:, class_idx]
        target_mask = (target == class_idx).float()
        
        # Only calculate dice if target class is present
        if target_mask.sum() > 0:
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            
            if union > 0:
                dice_loss += 1 - (2. * intersection + 1e-6) / (union + 1e-6)
                valid_classes += 1
    
    # Average over valid classes
    if valid_classes > 0:
        dice_loss = dice_loss / valid_classes
    else:
        dice_loss = torch.tensor(0.0, device=pred.device)
    
    return alpha * focal_loss + beta * dice_loss

def weighted_cross_entropy_loss(pred, target, class_weights=None):
    """
    Weighted cross entropy loss to handle class imbalance
    """
    target = target.squeeze(1)
    
    if class_weights is None:
        # Calculate class weights based on frequency
        class_counts = torch.bincount(target.view(-1), minlength=4)
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (4 * class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * 4  # Normalize
    
    return nn.CrossEntropyLoss(weight=class_weights)(pred, target)

def improved_metrics(pred, target):
    """
    Improved metric calculation with better handling of edge cases
    """
    target = target.squeeze(1)
    pred = torch.argmax(pred, dim=1)
    
    # Calculate per-class metrics
    dice_scores = []
    iou_scores = []
    
    for class_idx in range(1, 4):  # Skip background, include spleen, liver, kidneys
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        # Only calculate if target class is present
        if target_mask.sum() > 0:
            intersection = (pred_mask & target_mask).sum().float()
            union = pred_mask.sum() + target_mask.sum()
            
            if union > 0:
                dice = (2. * intersection + 1e-6) / (union + 1e-6)
                iou = intersection / (union - intersection + 1e-6)
                dice_scores.append(dice.item())
                iou_scores.append(iou.item())
    
    # Return average over classes that were present
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    avg_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    return avg_dice, avg_iou

def get_learning_rate_schedule(optimizer, initial_lr, epochs):
    """
    Create a learning rate schedule
    """
    def lr_lambda(epoch):
        if epoch < epochs * 0.3:
            return 1.0
        elif epoch < epochs * 0.7:
            return 0.1
        else:
            return 0.01
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def print_training_recommendations():
    """
    Print recommendations for fixing training issues
    """
    print("🔧 TRAINING FIX RECOMMENDATIONS")
    print("=" * 50)
    
    print("\n1. LEARNING RATE:")
    print("   - Reduce learning rate to 0.0001 or 0.0005")
    print("   - Use learning rate scheduling")
    print("   - Start with warmup for first 5 epochs")
    
    print("\n2. LOSS FUNCTION:")
    print("   - Use focal loss to handle class imbalance")
    print("   - Add class weights to cross entropy")
    print("   - Consider using only dice loss initially")
    
    print("\n3. DATA PREPROCESSING:")
    print("   - Check if labels are correctly mapped (0-3)")
    print("   - Verify image normalization")
    print("   - Ensure no NaN or infinite values")
    
    print("\n4. MODEL ARCHITECTURE:")
    print("   - Add batch normalization")
    print("   - Use dropout for regularization")
    print("   - Consider smaller model initially")
    
    print("\n5. TRAINING STRATEGY:")
    print("   - Start with smaller batch size (1-2)")
    print("   - Use gradient clipping")
    print("   - Monitor gradient norms")
    print("   - Check for exploding/vanishing gradients")
    
    print("\n6. CLASS IMBALANCE:")
    print("   - Most medical images are mostly background")
    print("   - Use focal loss or weighted loss")
    print("   - Consider data augmentation")
    print("   - Use smaller learning rate for background class")

if __name__ == "__main__":
    print_training_recommendations() 