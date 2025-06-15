import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import CombinedDataset, combined_transform
from utils.metrics import calculate_metrics, combined_loss
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

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def create_experiment_name(args):
    """Create a unique experiment name based on parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = f"bs{args.batch_size}_ep{args.epochs}_lr{args.lr}"
    return f"exp_{timestamp}_{param_str}"

def plot_training_metrics(log_file, save_dir):
    """Create and save plots of training metrics."""
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
    fig.suptitle('Training Metrics', fontsize=16)

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
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

    # Create a separate plot for training time
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, times, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_time.png'))
    plt.close()

def log_gpu_usage(log_file="gpu_usage.log"):
    with open(log_file, "a") as f:
        f.write(subprocess.getoutput("nvidia-smi"))
        f.write("\n" + "="*80 + "\n")

def train_one_epoch(model, loader, optimizer, accelerator, epoch, args):
    model.train()
    running_loss, total_iou, total_dice, total_acc = 0, 0, 0, 0
    num_batches = len(loader)

    # Only show progress bar on main process
    if accelerator.is_main_process:
        progress_bar = tqdm(loader, desc=f"🟢 Training Epoch {epoch+1}/{args.epochs}", leave=False)
        # Log GPU usage at start of training
        log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    else:
        progress_bar = loader

    for i, (images, labels) in enumerate(progress_bar):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, labels.float())
            accelerator.backward(loss)
            optimizer.step()

        # Calculate metrics
        outputs = torch.sigmoid(outputs)
        dice, iou, acc = calculate_metrics(outputs, labels)

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

def evaluate(model, loader, accelerator, epoch, args):
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
            loss = combined_loss(outputs, labels.float())
            outputs = torch.sigmoid(outputs)
            
            dice, iou, acc = calculate_metrics(outputs, labels)
            
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
        print(f"[START] 🚀 Starting Training\n" + "=" * 50)

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
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
        
        # Log initial GPU usage
        gpu_log_file = os.path.join(log_dir, 'gpu_usage.log')
        log_gpu_usage(gpu_log_file)

    # Datasets and loaders
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    test_dir = os.path.join(args.data_root, 'test')

    train_dataset = CombinedDataset(train_dir, transform=combined_transform)
    val_dataset = CombinedDataset(val_dir)
    test_dataset = CombinedDataset(test_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model and optimizer
    model = UNet3D(in_channels=1, out_channels=1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    # Log file setup
    if accelerator.is_main_process:
        log_file = os.path.join(log_dir, 'train_log.csv')
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'time', 'train_loss', 'val_loss', 
                            'train_dice', 'val_dice', 'train_iou', 'val_iou',
                            'train_acc', 'val_acc'])

    # Training loop
    best_val_dice = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Training
        train_loss, train_iou, train_dice, train_acc = train_one_epoch(
            model, train_loader, optimizer, accelerator, epoch, args)
        
        # Validation
        val_loss, val_iou, val_dice, val_acc = evaluate(
            model, val_loader, accelerator, epoch, args)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics only on main process
        if accelerator.is_main_process:
            print(f"[EPOCH] 📊 Epoch {epoch+1}/{args.epochs} - Time: {format_time(epoch_time)} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
                  f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, epoch_time, train_loss, val_loss,
                               train_dice, val_dice, train_iou, val_iou,
                               train_acc, val_acc])
            
            # Log GPU usage after each epoch
            log_gpu_usage(gpu_log_file)

        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_name = f"checkpoint_epoch{epoch+1}_{experiment_name}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            accelerator.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
            }, checkpoint_path)

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            best_model_path = os.path.join(checkpoint_dir, f"best_model_{experiment_name}.pth")
            accelerator.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
            }, best_model_path)

    # Plot training metrics only on main process
    if accelerator.is_main_process:
        plot_training_metrics(log_file, plots_dir)
        total_time = time.time() - start_time
        print(f"\n[END] ✅ Training completed in {format_time(total_time)}")
        print(f"Best validation Dice score: {best_val_dice:.4f}")
        
        # Log final GPU usage
        log_gpu_usage(gpu_log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet3D model')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Directory to save experiments')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
    
    args = parser.parse_args()
    main(args) 