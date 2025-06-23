#!/usr/bin/env python3
"""
Debug script to diagnose training issues
"""

import torch
import numpy as np
from utils.dataloader import CombinedDataset
from utils.metrics import combined_loss, calculate_dice, calculate_iou, calculate_accuracy
from models.unet import UNet3D
from torch.utils.data import DataLoader

def debug_dataset():
    """Debug dataset loading and preprocessing"""
    print("=== DATASET DEBUG ===")
    
    # Load a small dataset
    dataset = CombinedDataset('datasets/resampled/train', modalities=['ct'])
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("❌ No samples loaded! Check dataset path and modality filtering.")
        return
    
    # Check first few samples
    for i in range(min(3, len(dataset))):
        image, label = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Label shape: {label.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Label unique values: {torch.unique(label)}")
        print(f"  Label distribution: {torch.bincount(label.view(-1))}")
        
        # Check if labels are reasonable
        if label.max() > 3:
            print(f"  ⚠️  WARNING: Label max value {label.max()} > 3 (expected 0-3)")
        if label.min() < 0:
            print(f"  ⚠️  WARNING: Label min value {label.min()} < 0")

def debug_model_output():
    """Debug model output and loss calculation"""
    print("\n=== MODEL DEBUG ===")
    
    # Create a simple model
    model = UNet3D(in_channels=1, out_channels=4)
    model.eval()
    
    # Create dummy data
    batch_size = 2
    image = torch.randn(batch_size, 1, 64, 64, 64)  # Simple 3D volume
    label = torch.randint(0, 4, (batch_size, 1, 64, 64, 64))  # Random labels 0-3
    
    print(f"Input image shape: {image.shape}")
    print(f"Input label shape: {label.shape}")
    print(f"Label unique values: {torch.unique(label)}")
    
    # Get model output
    with torch.no_grad():
        output = model(image)
        print(f"Model output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check softmax
        softmax_output = torch.softmax(output, dim=1)
        print(f"Softmax range: [{softmax_output.min():.3f}, {softmax_output.max():.3f}]")
        print(f"Softmax sum per voxel: {softmax_output.sum(dim=1).min():.3f} - {softmax_output.sum(dim=1).max():.3f}")
        
        # Check predictions
        pred = torch.argmax(output, dim=1)
        print(f"Prediction unique values: {torch.unique(pred)}")
        print(f"Prediction distribution: {torch.bincount(pred.view(-1))}")
        
        # Calculate loss
        loss = combined_loss(output, label)
        print(f"Combined loss: {loss.item():.4f}")
        
        # Calculate metrics
        dice = calculate_dice(output, label)
        iou = calculate_iou(output, label)
        acc = calculate_accuracy(output, label)
        print(f"Dice: {dice:.4f}")
        print(f"IoU: {iou:.4f}")
        print(f"Accuracy: {acc:.4f}")

def debug_metric_calculation():
    """Debug metric calculation specifically"""
    print("\n=== METRIC DEBUG ===")
    
    # Create test cases
    test_cases = [
        ("Perfect prediction", torch.tensor([[[1]]]), torch.tensor([[[1]]])),
        ("All background", torch.tensor([[[0]]]), torch.tensor([[[0]]])),
        ("Random prediction", torch.randint(0, 4, (1, 1, 8, 8, 8)), torch.randint(0, 4, (1, 1, 8, 8, 8))),
    ]
    
    for name, pred, target in test_cases:
        print(f"\n{name}:")
        print(f"  Pred unique: {torch.unique(pred)}")
        print(f"  Target unique: {torch.unique(target)}")
        
        # Create model output format (B, C, H, W, D)
        pred_onehot = torch.zeros(1, 4, *pred.shape[2:])
        for i in range(4):
            pred_onehot[0, i] = (pred == i).float()
        
        dice = calculate_dice(pred_onehot, target)
        iou = calculate_iou(pred_onehot, target)
        acc = calculate_accuracy(pred_onehot, target)
        
        print(f"  Dice: {dice:.4f}")
        print(f"  IoU: {iou:.4f}")
        print(f"  Accuracy: {acc:.4f}")

def debug_loss_function():
    """Debug the loss function"""
    print("\n=== LOSS DEBUG ===")
    
    # Test different scenarios
    scenarios = [
        ("All background", torch.zeros(1, 4, 8, 8, 8), torch.zeros(1, 1, 8, 8, 8)),
        ("All class 1", torch.zeros(1, 4, 8, 8, 8), torch.ones(1, 1, 8, 8, 8)),
        ("Mixed classes", torch.zeros(1, 4, 8, 8, 8), torch.randint(0, 4, (1, 1, 8, 8, 8))),
    ]
    
    for name, pred, target in scenarios:
        print(f"\n{name}:")
        
        # Set some reasonable logits
        if "All class 1" in name:
            pred[0, 1] = 10.0  # High logit for class 1
        elif "Mixed classes" in name:
            pred[0, 0] = 1.0
            pred[0, 1] = 2.0
            pred[0, 2] = 3.0
            pred[0, 3] = 4.0
        
        loss = combined_loss(pred, target)
        print(f"  Loss: {loss.item():.4f}")

if __name__ == "__main__":
    print("🔍 DEBUGGING TRAINING ISSUES")
    print("=" * 50)
    
    try:
        debug_dataset()
        debug_model_output()
        debug_metric_calculation()
        debug_loss_function()
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("�� DEBUG COMPLETE") 