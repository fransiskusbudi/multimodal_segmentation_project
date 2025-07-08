#!/usr/bin/env python3
"""
Domain-Adversarial Neural Network (DANN) for Medical Image Segmentation
Adapts a model trained on source domain (e.g., MRI) to target domain (e.g., CT)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from utils.dataloader import CombinedDataset, combined_transform
from utils.metrics import calculate_metrics, combined_loss, calculate_iou, calculate_dice, calculate_accuracy, combined_ce_tversky_loss, distillation_loss
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

def create_experiment_name(args):
    """Create a unique experiment name based on parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = f"dann_{args.source_modality}_to_{args.target_modality}_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_wd{args.weight_decay}"
    if args.lambda_domain != 1.0:
        param_str += f"_lambda{args.lambda_domain}"
    return f"exp_{timestamp}_{param_str}"

def freeze_encoder(model):
    """Freeze the encoder layers of the UNet model."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.pool.parameters():
        param.requires_grad = False

def unfreeze_encoder(model):
    """Unfreeze the encoder layers of the UNet model."""
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.pool.parameters():
        param.requires_grad = True

def update_optimizer_for_frozen_encoder(model, optimizer, lr):
    """Update optimizer to only include unfrozen parameters."""
    # Get only the parameters that require gradients
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=optimizer.param_groups[0]['weight_decay'])
    return optimizer

def plot_training_metrics(log_file, save_dir):
    """Create and save plots of training metrics."""
    # Read the CSV file
    epochs, times, task_losses, domain_losses, val_losses, train_dices, val_dices, train_ious, val_ious, train_accs, val_accs, encoder_frozen = [], [], [], [], [], [], [], [], [], [], []
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            times.append(float(row['time']))
            task_losses.append(float(row['task_loss']))
            domain_losses.append(float(row['domain_loss']))
            val_losses.append(float(row['val_loss']))
            train_dices.append(float(row['train_dice']))
            val_dices.append(float(row['val_dice']))
            train_ious.append(float(row['train_iou']))
            val_ious.append(float(row['val_iou']))
            train_accs.append(float(row['train_acc']))
            val_accs.append(float(row['val_acc']))
            encoder_frozen.append(row.get('encoder_frozen', 'False').lower() == 'true')

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DANN Training Metrics', fontsize=16)

    # Plot losses
    ax1.plot(epochs, task_losses, label='Task Loss', marker='o')
    ax1.plot(epochs, domain_losses, label='Domain Loss', marker='o')
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

    # Add encoder freezing visualization to all subplots
    if any(encoder_frozen):
        for ax in [ax1, ax2, ax3, ax4]:
            # Find where encoder freezing changes
            frozen_regions = []
            start_epoch = None
            for i, frozen in enumerate(encoder_frozen):
                if frozen and start_epoch is None:
                    start_epoch = epochs[i]
                elif not frozen and start_epoch is not None:
                    frozen_regions.append((start_epoch, epochs[i-1]))
                    start_epoch = None
            if start_epoch is not None:
                frozen_regions.append((start_epoch, epochs[-1]))
            
            # Shade frozen regions
            for start, end in frozen_regions:
                ax.axvspan(start, end, alpha=0.2, color='red', label='Encoder Frozen' if start == frozen_regions[0][0] else "")
                ax.axvline(x=start, color='red', linestyle='--', alpha=0.7)
                ax.axvline(x=end, color='red', linestyle='--', alpha=0.7)
            
            if frozen_regions:
                ax.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dann_metrics.png'))
    plt.close()

    # Create a separate plot for training time
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, times, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('DANN Training Time per Epoch')
    plt.grid(True)

    # Add encoder freezing visualization
    if any(encoder_frozen):
        frozen_regions = []
        start_epoch = None
        for i, frozen in enumerate(encoder_frozen):
            if frozen and start_epoch is None:
                start_epoch = epochs[i]
            elif not frozen and start_epoch is not None:
                frozen_regions.append((start_epoch, epochs[i-1]))
                start_epoch = None
        if start_epoch is not None:
            frozen_regions.append((start_epoch, epochs[-1]))
        
        for start, end in frozen_regions:
            plt.axvspan(start, end, alpha=0.2, color='red', label='Encoder Frozen' if start == frozen_regions[0][0] else "")
            plt.axvline(x=start, color='red', linestyle='--', alpha=0.7)
            plt.axvline(x=end, color='red', linestyle='--', alpha=0.7)
        
        if frozen_regions:
            plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'dann_time.png'))
    plt.close()

def log_gpu_usage(log_file="gpu_usage.log"):
    with open(log_file, "a") as f:
        f.write(subprocess.getoutput("nvidia-smi"))
        f.write("\n" + "="*80 + "\n")

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for DANN
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for DANN
    """
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, alpha=1.0):
        # Apply gradient reversal
        reversed_features = self.grl(features, alpha)
        # Flatten features if needed
        if len(reversed_features.shape) > 2:
            reversed_features = reversed_features.view(reversed_features.size(0), -1)
        return self.classifier(reversed_features)

def dann_loss(task_loss, domain_loss, lambda_domain=1.0):
    """
    Combined DANN loss
    """
    return task_loss + lambda_domain * domain_loss

def train_one_epoch(model, domain_discriminator, source_loader, target_loader, 
                   task_optimizer, domain_optimizer, accelerator, epoch, args):
    """
    Train one epoch with DANN
    """
    model.train()
    domain_discriminator.train()
    
    running_task_loss, running_domain_loss, total_iou, total_dice, total_acc = 0, 0, 0, 0, 0
    num_batches = min(len(source_loader), len(target_loader))
    
    # Calculate alpha for gradient reversal (increases over time)
    alpha = 2.0 / (1.0 + np.exp(-10 * epoch / args.epochs)) - 1.0
    
    if accelerator.is_main_process:
        progress_bar = tqdm(zip(source_loader, target_loader), 
                           total=num_batches, 
                           desc=f"🟢 DANN Epoch {epoch+1}/{args.epochs}", 
                           leave=False)
    else:
        progress_bar = zip(source_loader, target_loader)
    
    for i, ((source_images, source_labels), (target_images, _)) in enumerate(progress_bar):
        if i >= num_batches:
            break
            
        with accelerator.accumulate(model):
            # Task-specific forward pass (segmentation)
            source_outputs = model(source_images)
            target_outputs = model(target_images)
            
            # Task loss (only on source domain where we have labels)
            task_loss = combined_ce_tversky_loss(source_outputs, source_labels)
            
            # Domain discrimination
            # Extract features from bottleneck layer
            source_features = model.get_bottleneck_features(source_images)
            target_features = model.get_bottleneck_features(target_images)
            
            # Create domain labels: 0 for source, 1 for target
            source_domain_labels = torch.zeros(source_features.size(0), 1, device=accelerator.device)
            target_domain_labels = torch.ones(target_features.size(0), 1, device=accelerator.device)
            
            # Domain discriminator forward pass
            source_domain_pred = domain_discriminator(source_features, alpha)
            target_domain_pred = domain_discriminator(target_features, alpha)
            
            # Domain loss
            domain_loss = nn.BCELoss()(source_domain_pred, source_domain_labels) + \
                         nn.BCELoss()(target_domain_pred, target_domain_labels)
            
            # Combined DANN loss
            total_loss = dann_loss(task_loss, domain_loss, args.lambda_domain)
            
            # Backward pass
            accelerator.backward(total_loss)
            task_optimizer.step()
            domain_optimizer.step()
            task_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            
            # Calculate metrics (only on source domain)
            with torch.no_grad():
                iou = calculate_iou(source_outputs, source_labels)
                dice = calculate_dice(source_outputs, source_labels)
                acc = calculate_accuracy(source_outputs, source_labels)
            
            # Gather metrics
            gathered_task_loss = accelerator.gather(torch.tensor(task_loss.item(), device=accelerator.device)).mean().item()
            gathered_domain_loss = accelerator.gather(torch.tensor(domain_loss.item(), device=accelerator.device)).mean().item()
            gathered_dice = accelerator.gather(torch.tensor(dice, device=accelerator.device)).mean().item()
            gathered_iou = accelerator.gather(torch.tensor(iou, device=accelerator.device)).mean().item()
            gathered_acc = accelerator.gather(torch.tensor(acc, device=accelerator.device)).mean().item()
            
            running_task_loss += gathered_task_loss
            running_domain_loss += gathered_domain_loss
            total_dice += gathered_dice
            total_iou += gathered_iou
            total_acc += gathered_acc
            
            if accelerator.is_main_process:
                progress_bar.set_postfix(
                    task_loss=gathered_task_loss, 
                    domain_loss=gathered_domain_loss,
                    dice=gathered_dice, 
                    iou=gathered_iou, 
                    acc=gathered_acc,
                    alpha=alpha
                )
                
                # Log GPU usage every 10 batches
                if i % 10 == 0:
                    log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    
    return (running_task_loss / num_batches,
            running_domain_loss / num_batches,
            total_iou / num_batches,
            total_dice / num_batches,
            total_acc / num_batches)

def evaluate(model, loader, accelerator, epoch, args):
    """
    Evaluate the model on validation set
    """
    model.eval()
    running_loss, total_iou, total_dice, total_acc = 0, 0, 0, 0
    num_batches = len(loader)
    
    if accelerator.is_main_process:
        progress_bar = tqdm(loader, desc=f"🔵 Validation Epoch {epoch+1}/{args.epochs}", leave=False)
        # Log GPU usage at start of validation
        log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    else:
        progress_bar = loader
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(progress_bar):
            outputs = model(images)
            loss = combined_ce_tversky_loss(outputs, labels)
            
            iou = calculate_iou(outputs, labels)
            dice = calculate_dice(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            
            gathered_loss = accelerator.gather(torch.tensor(loss.item(), device=accelerator.device)).mean()
            gathered_dice = accelerator.gather(torch.tensor(dice, device=accelerator.device)).mean()
            gathered_iou = accelerator.gather(torch.tensor(iou, device=accelerator.device)).mean()
            gathered_acc = accelerator.gather(torch.tensor(acc, device=accelerator.device)).mean()
            
            running_loss += gathered_loss.item()
            total_iou += gathered_iou.item()
            total_dice += gathered_dice.item()
            total_acc += gathered_acc.item()
            
            if accelerator.is_main_process:
                progress_bar.set_postfix(
                    val_loss=gathered_loss.item(), 
                    val_dice=gathered_dice.item(), 
                    val_iou=gathered_iou.item(), 
                    val_acc=gathered_acc.item()
                )
                
                # Log GPU usage every 10 batches
                if i % 10 == 0:
                    log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    
    return (running_loss / num_batches,
            total_iou / num_batches,
            total_dice / num_batches,
            total_acc / num_batches)

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
        print(f"[START] 🚀 Starting DANN Training\n" + "=" * 50)
        print(f"Source domain: {args.source_modality}")
        print(f"Target domain: {args.target_modality}")
        if args.freeze_encoder_epoch is not None:
            print(f"[INFO] 🔒 Encoder will be frozen at epoch {args.freeze_encoder_epoch}")

    # Create unique experiment directory
    experiment_name = create_experiment_name(args)
    args.experiment_name = experiment_name  # Store experiment name in args for GPU logging
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
            f.write(f"DANN Configuration:\n")
            f.write(f"Source modality: {args.source_modality}\n")
            f.write(f"Target modality: {args.target_modality}\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
        
        # Log initial GPU usage
        gpu_log_file = os.path.join(log_dir, 'gpu_usage.log')
        log_gpu_usage(gpu_log_file)

    # Datasets and loaders
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    test_dir = os.path.join(args.data_root, 'test')
    
    # Source domain dataset (with labels)
    source_dataset = CombinedDataset(train_dir, transform=combined_transform(), modalities=[args.source_modality])
    # Target domain dataset (without labels for training)
    target_dataset = CombinedDataset(train_dir, transform=combined_transform(), modalities=[args.target_modality])
    # Validation dataset (use source domain for validation)
    val_dataset = CombinedDataset(val_dir, modalities=[args.source_modality])
    test_dataset = CombinedDataset(test_dir, modalities=[args.source_modality])
    
    # Subsample if specified
    if args.n_samples is not None:
        rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()
        source_indices = rng.choice(len(source_dataset), size=args.n_samples, replace=False)
        target_indices = rng.choice(len(target_dataset), size=args.n_samples, replace=False)
        source_dataset = Subset(source_dataset, source_indices)
        target_dataset = Subset(target_dataset, target_indices)
    
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Models
    model = UNet3D(in_channels=1, out_channels=4, dropout_rate=args.dropout_rate)  # 4 classes: background + spleen + liver + kidneys
    domain_discriminator = DomainDiscriminator(input_dim=512, hidden_dim=256)  # Adjust input_dim based on your bottleneck features
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        if accelerator.is_main_process:
            print(f"Loading pre-trained model from: {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=accelerator.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    # Optimizers
    task_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    domain_optimizer = optim.AdamW(domain_discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Add ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(task_optimizer, mode='max', patience=10, factor=0.1, min_lr=1e-6, verbose=True)
    
    # Prepare for distributed training
    model, domain_discriminator, task_optimizer, domain_optimizer, scheduler, source_loader, target_loader, val_loader, test_loader = accelerator.prepare(
        model, domain_discriminator, task_optimizer, domain_optimizer, scheduler, source_loader, target_loader, val_loader, test_loader
    )

    # Log file setup
    log_file = os.path.join(log_dir, 'dann_log.csv')
    if accelerator.is_main_process:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'time', 'task_loss', 'domain_loss', 'val_loss', 
                           'train_dice', 'val_dice', 'train_iou', 'val_iou',
                           'train_acc', 'val_acc', 'encoder_frozen'])

    # Training loop
    best_val_dice = 0
    start_time = time.time()
    encoder_frozen = False
    # Early stopping variables
    patience_counter = 0
    early_stop = False
    
    for epoch in range(args.epochs):
        if early_stop:
            break
        epoch_start_time = time.time()
        
        # Check if we need to freeze/unfreeze encoder
        if args.freeze_encoder_epoch is not None:
            if epoch == args.freeze_encoder_epoch and not encoder_frozen:
                # Freeze encoder at specified epoch
                if accelerator.is_main_process:
                    print(f"[INFO] 🔒 Freezing encoder at epoch {epoch+1}")
                freeze_encoder(model)
                task_optimizer = update_optimizer_for_frozen_encoder(model, task_optimizer, args.lr)
                # Re-prepare optimizer with accelerator
                task_optimizer = accelerator.prepare(task_optimizer)
                encoder_frozen = True
            elif epoch == args.freeze_encoder_epoch + 1 and encoder_frozen:
                # Unfreeze encoder after one epoch
                if accelerator.is_main_process:
                    print(f"[INFO] 🔓 Unfreezing encoder at epoch {epoch+1}")
                unfreeze_encoder(model)
                task_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                # Re-prepare optimizer with accelerator
                task_optimizer = accelerator.prepare(task_optimizer)
                encoder_frozen = False
        
        # Training
        task_loss, domain_loss, train_iou, train_dice, train_acc = train_one_epoch(
            model, domain_discriminator, source_loader, target_loader,
            task_optimizer, domain_optimizer, accelerator, epoch, args
        )
        
        # Validation
        val_loss, val_iou, val_dice, val_acc = evaluate(
            model, val_loader, accelerator, epoch, args)
        
        # Step the scheduler with validation loss
        scheduler.step(val_dice)
        
        # Log learning rate
        if accelerator.is_main_process:
            current_lr = task_optimizer.param_groups[0]['lr']
            print(f"[LR] Learning rate after epoch {epoch+1}: {current_lr}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics only on main process
        if accelerator.is_main_process:
            print(f"[EPOCH] 📊 DANN Epoch {epoch+1}/{args.epochs} - Time: {format_time(epoch_time)} | "
                  f"Task Loss: {task_loss:.4f} | Domain Loss: {domain_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
                  f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Encoder: {'🔒' if encoder_frozen else '🔓'}")
            
            # Save to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, epoch_time, task_loss, domain_loss, val_loss,
                               train_dice, val_dice, train_iou, val_iou,
                               train_acc, val_acc, encoder_frozen])

            # Log GPU usage after each epoch
            log_gpu_usage(gpu_log_file)

        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_discriminator = accelerator.unwrap_model(domain_discriminator)
            checkpoint_name = f"checkpoint_epoch{epoch+1}_{experiment_name}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            accelerator.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'discriminator_state_dict': unwrapped_discriminator.state_dict(),
                'task_optimizer_state_dict': task_optimizer.state_dict(),
                'domain_optimizer_state_dict': domain_optimizer.state_dict(),
                'task_loss': task_loss,
                'domain_loss': domain_loss,
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
            unwrapped_discriminator = accelerator.unwrap_model(domain_discriminator)
            best_model_path = os.path.join(checkpoint_dir, f"best_dann_model_{experiment_name}.pth")
            accelerator.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'discriminator_state_dict': unwrapped_discriminator.state_dict(),
                'task_optimizer_state_dict': task_optimizer.state_dict(),
                'domain_optimizer_state_dict': domain_optimizer.state_dict(),
                'task_loss': task_loss,
                'domain_loss': domain_loss,
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
                        print(f"[EARLY STOPPING] No improvement in validation Dice for {args.patience} epochs. Stopping training early at epoch {epoch+1}.")
                    early_stop = True

    # Plot training metrics only on main process
    if accelerator.is_main_process:
        plot_training_metrics(log_file, plots_dir)
        total_time = time.time() - start_time
        print(f"\n[END] ✅ DANN Training completed in {format_time(total_time)}")
        print(f"Best validation Dice score: {best_val_dice:.4f}")
        
        # Log final GPU usage
        log_gpu_usage(gpu_log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DANN for Medical Image Segmentation')
    parser.add_argument('--source_modality', type=str, required=True, help='Source domain modality (e.g., mri)')
    parser.add_argument('--target_modality', type=str, required=True, help='Target domain modality (e.g., ct)')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Directory to save experiments')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pre-trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('--lambda_domain', type=float, default=1.0, help='Weight for domain loss')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to use (for ablation study)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('--freeze_encoder_epoch', type=int, default=None, help='Epoch to freeze the encoder')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping based on validation Dice')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait for improvement before stopping (used if early stopping is enabled)')
    
    args = parser.parse_args()
    main(args) 