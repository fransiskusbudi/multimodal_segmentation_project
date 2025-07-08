#!/usr/bin/env python3
"""
Fine-tuning script for CT data
Loads a pre-trained model and continues training on CT-specific data
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from utils.dataloader import CombinedDataset, combined_transform
from utils.metrics import calculate_metrics, combined_loss, calculate_iou, calculate_dice, calculate_accuracy, tversky_loss, combined_ce_tversky_loss
from models.unet import UNet3D
import numpy as np
import csv
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.utils import set_seed
import subprocess
import torch.nn.functional as F
import json

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def create_finetune_experiment_name(args):
    """Create a unique experiment name for fine-tuning."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model = os.path.basename(args.pretrained_model).split('.pth')[0]
    freeze_str = "frozen_enc" if args.freeze_encoder else "unfrozen_enc"
    param_str = f"ft_ct_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_wd{args.weight_decay}_{freeze_str}"
    return f"finetune_{timestamp}_{base_model}_samples_{args.n_samples}"#_{param_str}"

def plot_finetune_metrics(log_file, save_dir):
    """Create and save plots of fine-tuning metrics."""
    # Read the CSV file
    epochs, times, train_losses, val_losses, train_dices, val_dices, train_ious, val_ious, train_accs, val_accs = [], [], [], [], [], [], [], [], [], []
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            times.append(float(row['time']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            train_dices.append(float(row['train_dice']))
            val_dices.append(float(row['val_dice']))
            train_ious.append(float(row['train_iou']))
            val_ious.append(float(row['val_iou']))
            train_accs.append(float(row['train_acc']))
            val_accs.append(float(row['val_acc']))

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fine-tuning Metrics (CT Data)', fontsize=16)

    # Plot losses
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Dice scores
    ax2.plot(epochs, train_dices, label='Train Dice', marker='o')
    ax2.plot(epochs, val_dices, label='Val Dice', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Training and Validation Dice Score')
    ax2.legend()
    ax2.grid(True)

    # Plot IoU scores
    ax3.plot(epochs, train_ious, label='Train IoU', marker='o')
    ax3.plot(epochs, val_ious, label='Val IoU', marker='o')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('IoU Score')
    ax3.set_title('Training and Validation IoU Score')
    ax3.legend()
    ax3.grid(True)

    # Plot accuracy
    ax4.plot(epochs, train_accs, label='Train Acc', marker='o')
    ax4.plot(epochs, val_accs, label='Val Acc', marker='o')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Training and Validation Accuracy')
    ax4.legend()
    ax4.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetune_metrics.png'))
    plt.close()

    # Create a separate plot for training time
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, times, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Fine-tuning Time per Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'finetune_time.png'))
    plt.close()

def log_gpu_usage(log_file="gpu_usage.log"):
    with open(log_file, "a") as f:
        f.write(subprocess.getoutput("nvidia-smi"))
        f.write("\n" + "="*80 + "\n")

def get_loss_fn(loss_type):
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'tversky':
        def loss_fn(pred, target):
            return tversky_loss(pred, target, alpha=0.5, beta=0.5)
        return loss_fn
    elif loss_type == 'dice':
        def loss_fn(pred, target):
            target_ = target.squeeze(1)
            pred_softmax = F.softmax(pred, dim=1)
            dice = 0
            for class_idx in range(1, pred_softmax.size(1)):
                pred_mask = pred_softmax[:, class_idx]
                target_mask = (target_ == class_idx).float()
                intersection = (pred_mask * target_mask).sum()
                union = pred_mask.sum() + target_mask.sum()
                dice += 1 - (2. * intersection + 1e-5) / (union + 1e-5)
            dice = dice / (pred_softmax.size(1) - 1)
            return dice
        return loss_fn
    elif loss_type == 'ce_tversky':
        def loss_fn(pred, target):
            return combined_ce_tversky_loss(pred, target, alpha=0.5, beta=0.5)
        return loss_fn
    else:
        return combined_loss

def train_one_epoch(model, loader, optimizer, accelerator, epoch, args, loss_fn):
    model.train()
    running_loss, total_iou, total_dice, total_acc = 0, 0, 0, 0
    num_batches = len(loader)

    # Only show progress bar on main process
    if accelerator.is_main_process:
        progress_bar = tqdm(loader, desc=f"🟢 Fine-tuning Epoch {epoch+1}/{args.epochs}", leave=False)
        # Log GPU usage at start of training
        log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    else:
        progress_bar = loader

    for i, (images, labels) in enumerate(progress_bar):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                iou = calculate_iou(outputs, labels)
                dice = calculate_dice(outputs, labels)
                acc = calculate_accuracy(outputs, labels)

            # Wrap scalar values as tensors for gathering
            gathered_loss = accelerator.gather(torch.tensor(loss.item(), device=accelerator.device)).mean().item()
            gathered_dice = accelerator.gather(torch.tensor(dice, device=accelerator.device)).mean().item()
            gathered_iou = accelerator.gather(torch.tensor(iou, device=accelerator.device)).mean().item()
            gathered_acc = accelerator.gather(torch.tensor(acc, device=accelerator.device)).mean().item()

            # Accumulate gathered values
            running_loss += gathered_loss
            total_dice += gathered_dice
            total_iou += gathered_iou
            total_acc += gathered_acc

            # Only update progress bar on main process
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=gathered_loss, iou=gathered_iou, dice=gathered_dice, acc=gathered_acc)
                
                # Log GPU usage every 10 batches
                if i % 10 == 0:
                    log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))

    return (running_loss / num_batches,
            total_iou / num_batches,
            total_dice / num_batches,
            total_acc / num_batches)

def evaluate(model, loader, accelerator, epoch, args, loss_fn):
    model.eval()
    running_loss, total_iou, total_dice, total_acc = 0, 0, 0, 0
    num_batches = len(loader)
    
    # Only show progress bar on main process
    if accelerator.is_main_process:
        progress_bar = tqdm(loader, desc=f"🔵 Validation Epoch {epoch+1}/{args.epochs}", leave=False)
        # Log GPU usage at start of validation
        log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    else:
        progress_bar = loader
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(progress_bar):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Calculate metrics
            iou = calculate_iou(outputs, labels)
            dice = calculate_dice(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            
            # Gather metrics from all processes
            gathered_loss = accelerator.gather(torch.tensor(loss.item(), device=accelerator.device)).mean()
            gathered_dice = accelerator.gather(torch.tensor(dice, device=accelerator.device)).mean()
            gathered_iou = accelerator.gather(torch.tensor(iou, device=accelerator.device)).mean()
            gathered_acc = accelerator.gather(torch.tensor(acc, device=accelerator.device)).mean()
            
            running_loss += gathered_loss.item()
            total_iou += gathered_iou.item()
            total_dice += gathered_dice.item()
            total_acc += gathered_acc.item()
        
            # Only update progress bar on main process
            if accelerator.is_main_process:
                progress_bar.set_postfix(val_loss=gathered_loss.item(), val_iou=gathered_iou.item(), 
                                       val_dice=gathered_dice.item(), val_acc=gathered_acc.item())
                
                # Log GPU usage every 10 batches
                if i % 10 == 0:
                    log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))

    return (running_loss / num_batches,
            total_iou / num_batches,
            total_dice / num_batches,
            total_acc / num_batches)

def load_pretrained_model(model_path, model, accelerator):
    """Load pre-trained model weights."""
    if accelerator.is_main_process:
        print(f"Loading pre-trained model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=accelerator.device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=True)
    
    if accelerator.is_main_process:
        print("✅ Pre-trained model loaded successfully!")
    
    return model

def freeze_encoder(model, accelerator):
    real_model = model.module if hasattr(model, "module") else model
    if accelerator.is_main_process:
        print("🔒 Freezing encoder layers to prevent overfitting...")
    for param in real_model.encoder.parameters():
        param.requires_grad = False
    for param in real_model.bottleneck.parameters():
        param.requires_grad = False
    frozen_params = sum(p.numel() for p in real_model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
    total_params = frozen_params + trainable_params
    if accelerator.is_main_process:
        print(f"📊 Parameter Summary:")
        print(f"   Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"   Total parameters: {total_params:,}")
        print("✅ Encoder frozen successfully!")

def unfreeze_encoder(model, accelerator):
    real_model = model.module if hasattr(model, "module") else model
    if accelerator.is_main_process:
        print("🔓 Unfreezing encoder layers...")
    for param in real_model.encoder.parameters():
        param.requires_grad = True
    for param in real_model.bottleneck.parameters():
        param.requires_grad = True
    frozen_params = sum(p.numel() for p in real_model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
    total_params = frozen_params + trainable_params
    if accelerator.is_main_process:
        print(f"📊 Parameter Summary:")
        print(f"   Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"   Total parameters: {total_params:,}")
        print("✅ Encoder unfrozen successfully!")

def main(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Only print on main process
    if accelerator.is_main_process:
        print(f"[START] 🚀 Starting CT Fine-tuning\n" + "=" * 50)

    # Create unique experiment directory
    experiment_name = create_finetune_experiment_name(args)
    args.experiment_name = experiment_name
    experiment_dir = os.path.join(args.experiment_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    plots_dir = os.path.join(experiment_dir, 'plots')

    # Create directories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Save experiment configuration
    if accelerator.is_main_process:
        config_file = os.path.join(experiment_dir, 'config.txt')
        with open(config_file, 'w') as f:
            f.write(f"Fine-tuning Configuration:\n")
            f.write(f"Pre-trained model: {args.pretrained_model}\n")
            f.write(f"CT data root: {args.data_root}\n")
            f.write(f"Encoder frozen: {args.freeze_encoder}\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
        
        # Log initial GPU usage
        gpu_log_file = os.path.join(log_dir, 'gpu_usage.log')
        log_gpu_usage(gpu_log_file)

    # Datasets and loaders - focus on CT data
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    test_dir = os.path.join(args.data_root, 'test')

    train_dataset = CombinedDataset(train_dir, transform=None, modalities=args.modalities)
    val_dataset = CombinedDataset(val_dir, modalities=args.modalities)
    test_dataset = CombinedDataset(test_dir, modalities=args.modalities)

    # Single-run ablation: sample n_samples if specified
    if args.n_samples is not None:
        rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()
        indices = rng.choice(len(train_dataset), size=args.n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model and optimizer
    model = UNet3D(in_channels=1, out_channels=4, dropout_rate=args.dropout_rate)  # 4 classes: background + spleen + liver + kidneys
    
    # Load pre-trained model
    model = load_pretrained_model(args.pretrained_model, model, accelerator)
    
    # Freeze encoder if requested (at start)
    encoder_frozen = False
    if args.freeze_encoder:
        freeze_encoder(model, accelerator)
        encoder_frozen = True
    # Optimizer - only optimize trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    
    # Log file setup
    if accelerator.is_main_process:
        log_file = os.path.join(log_dir, 'finetune_log.csv')
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'time', 'train_loss', 'val_loss', 
                           'train_dice', 'val_dice', 'train_iou', 'val_iou',
                           'train_acc', 'val_acc', 'encoder_frozen'])
    
    # Fine-tuning loop
    best_val_dice = 0
    start_time = time.time()
    # Early stopping variables
    patience_counter = 0
    early_stop = False
    
    loss_fn = get_loss_fn(args.loss)
    
    for epoch in range(args.epochs):
        if early_stop:
            break
        epoch_start_time = time.time()
        # Check if we need to freeze/unfreeze encoder at specific epoch
        if args.freeze_encoder_epoch is not None:
            if epoch == args.freeze_encoder_epoch and not encoder_frozen:
                if accelerator.is_main_process:
                    print(f"[INFO] 🔒 Freezing encoder at epoch {epoch+1}")
                freeze_encoder(model, accelerator)
                trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
                optimizer = accelerator.prepare(optimizer)
                encoder_frozen = True
            elif epoch == args.freeze_encoder_epoch + 1 and encoder_frozen:
                if accelerator.is_main_process:
                    print(f"[INFO] 🔓 Unfreezing encoder at epoch {epoch+1}")
                unfreeze_encoder(model, accelerator)
                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                optimizer = accelerator.prepare(optimizer)
                encoder_frozen = False
        # Training
        train_loss, train_iou, train_dice, train_acc = train_one_epoch(
            model, train_loader, optimizer, accelerator, epoch, args, loss_fn)
        # Validation
        val_loss, val_iou, val_dice, val_acc = evaluate(
            model, val_loader, accelerator, epoch, args, loss_fn)
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        # Log metrics only on main process
        if accelerator.is_main_process:
            print(f"[EPOCH] 📊 Fine-tuning Epoch {epoch+1}/{args.epochs} - Time: {format_time(epoch_time)} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
                  f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Encoder: {'🔒' if encoder_frozen else '🫠🔓'}")
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, epoch_time, train_loss, val_loss,
                               train_dice, val_dice, train_iou, val_iou,
                               train_acc, val_acc, encoder_frozen])
            log_gpu_usage(gpu_log_file)
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_name = f"finetune_checkpoint_epoch{epoch+1}_{experiment_name}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            accelerator.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
                'encoder_frozen': encoder_frozen,
            }, checkpoint_path)
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            best_model_path = os.path.join(checkpoint_dir, f"best_finetuned_model_{experiment_name}.pth")
            accelerator.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
                'encoder_frozen': encoder_frozen,
            }, best_model_path)
        else:
            if args.early_stopping:
                patience_counter += 1
                if patience_counter >= args.patience:
                    if accelerator.is_main_process:
                        print(f"[EARLY STOPPING] No improvement in validation Dice for {args.patience} epochs. Stopping fine-tuning early at epoch {epoch+1}.")
                    early_stop = True

    # Plot fine-tuning metrics only on main process
    if accelerator.is_main_process:
        plot_finetune_metrics(log_file, plots_dir)
        total_time = time.time() - start_time
        print(f"\n[END] ✅ CT Fine-tuning completed in {format_time(total_time)}")
        print(f"Best validation Dice score: {best_val_dice:.4f}")
        # Log final GPU usage
        log_gpu_usage(gpu_log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune UNet3D model on CT data')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to pre-trained model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Directory to save experiments')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for fine-tuning')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--modalities', type=str, default='ct', help='Comma-separated list of modalities to include')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder layers to prevent overfitting to CT data')
    parser.add_argument('--freeze_encoder_epoch', type=int, default=None, help='Epoch to freeze the encoder (set to null or comment out to disable)')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping based on validation Dice')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait for improvement before stopping (used if early stopping is enabled)')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for regularization (default: 0.1)')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to use for ablation study')
    parser.add_argument('--loss', type=str, default='ce_tversky', choices=['combined', 'ce', 'dice', 'tversky', 'ce_tversky'], help='Loss function to use')
    
    args = parser.parse_args()
    
    # Process modalities argument
    if args.modalities.lower() == 'all':
        args.modalities = None  # None means include all modalities
    else:
        args.modalities = [mod.strip().lower() for mod in args.modalities.split(',')]
    
    loss_fn = get_loss_fn(args.loss)
    
    main(args) 