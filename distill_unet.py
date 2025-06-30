import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import csv
from accelerate import Accelerator
from torch.utils.data import DataLoader
from models.unet import UNet3D
from utils.dataloader import CombinedDataset
from utils.metrics import combined_loss, calculate_dice, calculate_iou, calculate_accuracy
import matplotlib.pyplot as plt
import subprocess
from datetime import timedelta


def load_teacher_model(teacher_path, model, accelerator):
    if teacher_path is not None and os.path.exists(teacher_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % accelerator.process_index}
        checkpoint = torch.load(teacher_path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        if accelerator.is_main_process:
            print(f"Loaded teacher model from {teacher_path}")
    else:
        raise FileNotFoundError(f"Teacher model checkpoint not found at {teacher_path}")
    return model

def distillation_loss(student_logits, teacher_logits, target, alpha=0.7, temperature=2.0):
    # Standard segmentation loss (CE + Dice)
    seg_loss = combined_loss(student_logits, target)
    # KL divergence between teacher and student softmax outputs
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction='none')  # shape: (B, C, ...)
    kl_loss = kl_loss.mean() * (temperature ** 2)  # average over all elements
    return alpha * seg_loss + (1 - alpha) * kl_loss

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def log_gpu_usage(log_file="gpu_usage.log"):
    with open(log_file, "a") as f:
        f.write(subprocess.getoutput("nvidia-smi"))
        f.write("\n" + "="*80 + "\n")

def plot_distill_metrics(log_file, save_dir):
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distillation Metrics', fontsize=16)
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, train_dices, label='Train Dice', marker='o')
    ax2.plot(epochs, val_dices, label='Val Dice', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Training and Validation Dice Score')
    ax2.legend()
    ax2.grid(True)
    ax3.plot(epochs, train_ious, label='Train IoU', marker='o')
    ax3.plot(epochs, val_ious, label='Val IoU', marker='o')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('IoU Score')
    ax3.set_title('Training and Validation IoU Score')
    ax3.legend()
    ax3.grid(True)
    ax4.plot(epochs, train_accs, label='Train Acc', marker='o')
    ax4.plot(epochs, val_accs, label='Val Acc', marker='o')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Training and Validation Accuracy')
    ax4.legend()
    ax4.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distill_metrics.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, times, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Distillation Time per Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'distill_time.png'))
    plt.close()

def train_one_epoch(student, teacher, loader, optimizer, accelerator, epoch, args):
    student.train()
    teacher.eval()
    running_loss, total_iou, total_dice, total_acc = 0, 0, 0, 0
    num_batches = len(loader)
    progress_bar = loader
    if accelerator.is_main_process:
        from tqdm import tqdm
        progress_bar = tqdm(loader, desc=f"🟢 Distill Train Epoch {epoch+1}/{args.epochs}", leave=False)
        # Log GPU usage at start of training
        log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    for i, (images, labels) in enumerate(progress_bar):
        with accelerator.accumulate(student):
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            loss = distillation_loss(student_logits, teacher_logits, labels)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        # Metrics
        iou = calculate_iou(student_logits, labels)
        dice = calculate_dice(student_logits, labels)
        acc = calculate_accuracy(student_logits, labels)
        gathered_loss = accelerator.gather(torch.tensor(loss.item(), device=accelerator.device)).mean()
        gathered_dice = accelerator.gather(torch.tensor(dice, device=accelerator.device)).mean()
        gathered_iou = accelerator.gather(torch.tensor(iou, device=accelerator.device)).mean()
        gathered_acc = accelerator.gather(torch.tensor(acc, device=accelerator.device)).mean()
        running_loss += gathered_loss.item()
        total_iou += gathered_iou.item()
        total_dice += gathered_dice.item()
        total_acc += gathered_acc.item()
        # Update progress bar with metrics
        if accelerator.is_main_process:
            progress_bar.set_postfix(loss=gathered_loss.item(), iou=gathered_iou.item(), dice=gathered_dice.item(), acc=gathered_acc.item())
            # Log GPU usage every 10 batches
            if i % 10 == 0:
                log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    return running_loss / num_batches, total_iou / num_batches, total_dice / num_batches, total_acc / num_batches

def evaluate(student, loader, accelerator, epoch, args):
    student.eval()
    running_loss, total_iou, total_dice, total_acc = 0, 0, 0, 0
    num_batches = len(loader)
    progress_bar = loader
    if accelerator.is_main_process:
        from tqdm import tqdm
        progress_bar = tqdm(loader, desc=f"🔵 Distill Val Epoch {epoch+1}/{args.epochs}", leave=False)
        # Log GPU usage at start of validation
        log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    with torch.no_grad():
        for i, (images, labels) in enumerate(progress_bar):
            logits = student(images)
            loss = combined_loss(logits, labels)
            iou = calculate_iou(logits, labels)
            dice = calculate_dice(logits, labels)
            acc = calculate_accuracy(logits, labels)
            gathered_loss = accelerator.gather(torch.tensor(loss.item(), device=accelerator.device)).mean()
            gathered_dice = accelerator.gather(torch.tensor(dice, device=accelerator.device)).mean()
            gathered_iou = accelerator.gather(torch.tensor(iou, device=accelerator.device)).mean()
            gathered_acc = accelerator.gather(torch.tensor(acc, device=accelerator.device)).mean()
            running_loss += gathered_loss.item()
            total_iou += gathered_iou.item()
            total_dice += gathered_dice.item()
            total_acc += gathered_acc.item()
            # Update progress bar with metrics
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=gathered_loss.item(), iou=gathered_iou.item(), dice=gathered_dice.item(), acc=gathered_acc.item())
                # Log GPU usage every 10 batches
                if i % 10 == 0:
                    log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    return running_loss / num_batches, total_iou / num_batches, total_dice / num_batches, total_acc / num_batches

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    if accelerator.is_main_process:
        print(f"[START] 🚀 Starting Knowledge Distillation\n" + "=" * 50)
    # Create unique experiment directory
    experiment_name = f"distill_{time.strftime('%Y%m%d_%H%M%S')}"
    args.experiment_name = experiment_name
    experiment_dir = os.path.join(args.experiment_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    # Save config
    if accelerator.is_main_process:
        config_file = os.path.join(experiment_dir, 'config.txt')
        with open(config_file, 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
        # Log initial GPU usage
        gpu_log_file = os.path.join(log_dir, 'gpu_usage.log')
        log_gpu_usage(gpu_log_file)
    # Data
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    train_dataset = CombinedDataset(train_dir, modalities=args.modalities)
    val_dataset = CombinedDataset(val_dir, modalities=args.modalities)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    # Models
    teacher = UNet3D(in_channels=1, out_channels=4)
    student = UNet3D(in_channels=1, out_channels=4)
    teacher = load_teacher_model(args.teacher_model, teacher, accelerator)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(accelerator.device)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Accelerate
    student, optimizer, train_loader, val_loader = accelerator.prepare(student, optimizer, train_loader, val_loader)
    # Logging
    if accelerator.is_main_process:
        log_file = os.path.join(log_dir, 'distill_log.csv')
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'time', 'train_loss', 'val_loss', 'train_dice', 'val_dice', 'train_iou', 'val_iou', 'train_acc', 'val_acc'])
    best_val_dice = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss, train_iou, train_dice, train_acc = train_one_epoch(student, teacher, train_loader, optimizer, accelerator, epoch, args)
        val_loss, val_iou, val_dice, val_acc = evaluate(student, val_loader, accelerator, epoch, args)
        epoch_time = time.time() - epoch_start_time
        if accelerator.is_main_process:
            print(f"[EPOCH] 📊 Distill Epoch {epoch+1}/{args.epochs} - Time: {format_time(epoch_time)} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
                  f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, epoch_time, train_loss, val_loss, train_dice, val_dice, train_iou, val_iou, train_acc, val_acc])
            log_gpu_usage(gpu_log_file)
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({'model_state_dict': student.state_dict()}, os.path.join(checkpoint_dir, 'best_student.pth'))
    if accelerator.is_main_process:
        plot_distill_metrics(log_file, plots_dir)
        total_time = time.time() - start_time
        print(f"\n[END] ✅ Distillation completed in {format_time(total_time)}")
        print(f"Best validation Dice score: {best_val_dice:.4f}")
        log_gpu_usage(gpu_log_file)

def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge Distillation for 3D U-Net Segmentation')
    parser.add_argument('--teacher_model', type=str, required=True, help='Path to pre-trained teacher model checkpoint')
    parser.add_argument('--data_root', type=str, default='datasets/resampled', help='Root directory of dataset splits')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Directory to save experiments')
    parser.add_argument('--modalities', type=str, default='all', help='Comma-separated list of modalities to include (e.g., "ct", "mri", "ct,mri", "all")')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Mixed precision training type')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.modalities.lower() == 'all':
        args.modalities = None  # None means include all modalities
    else:
        args.modalities = [mod.strip().lower() for mod in args.modalities.split(',')]
    main(args) 