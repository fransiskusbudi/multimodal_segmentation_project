import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from utils.dataloader import CombinedDataset, combined_transform
from utils.metrics import calculate_metrics, combined_loss, calculate_iou, calculate_dice, calculate_accuracy, tversky_loss, combined_ce_tversky_loss
from models.unet_dann import UNet3D
import numpy as np
import csv
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
# Remove accelerate imports
# from accelerate import Accelerator
# from accelerate.utils import set_seed
import subprocess

# ---- DANN-specific components ----
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_):
    return GradientReversal.apply(x, lambda_)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

# ---- Utility functions (from train_unet.py) ----
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def create_experiment_name(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = f"bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_wd{args.weight_decay}_ld{args.lambda_domain}_add{args.n_add_source}_ns{args.n_samples}"
    return f"dann_{timestamp}_{param_str}"

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
            pred_softmax = torch.nn.functional.softmax(pred, dim=1)
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

def log_gpu_usage(log_file="gpu_usage.log"):
    with open(log_file, "a") as f:
        f.write(subprocess.getoutput("nvidia-smi"))
        f.write("\n" + "="*80 + "\n")

def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.pool.parameters():
        param.requires_grad = False

def unfreeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.pool.parameters():
        param.requires_grad = True

def update_optimizer_for_frozen_encoder(model, optimizer, lr):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=optimizer.param_groups[0]['weight_decay'])
    return optimizer

def plot_training_metrics(log_file, save_dir):
    """Create and save plots of training metrics (identical to train_unet.py style)."""
    # Read the CSV file
    epochs, times, train_losses, val_losses, train_dices, val_dices, train_ious, val_ious, train_accs, val_accs, encoder_frozen = [], [], [], [], [], [], [], [], [], [], []
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
            encoder_frozen.append(row.get('encoder_frozen', 'False').lower() == 'true')

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
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

    # Create a separate plot for training time
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, times, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
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
    plt.savefig(os.path.join(save_dir, 'training_time.png'))
    plt.close()

# ---- DANN Training Loop ----
def train_one_epoch_dann(seg_model, disc_model, loaders, task_optimizer, domain_optimizer, device, epoch, args, loss_fn, lambda_domain, scaler=None, use_fp16=False):
    seg_model.train()
    disc_model.train()
    source_loader, target_loader = loaders
    running_task_loss, running_domain_loss, total_iou, total_dice, total_acc = 0, 0, 0, 0, 0
    num_batches = min(len(source_loader), len(target_loader))
    progress_bar = tqdm(zip(source_loader, target_loader), total=num_batches, desc=f"🟢 DANN Epoch {epoch+1}/{args.epochs}", leave=False)
    grad_accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    for batch_idx, ((source_images, source_labels), (target_images, _)) in enumerate(progress_bar):
        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)
        if batch_idx % grad_accum_steps == 0:
            task_optimizer.zero_grad()
            domain_optimizer.zero_grad()
        if use_fp16:
            with torch.cuda.amp.autocast():
                # Segmentation (task) loss
                source_outputs, source_features = seg_model(source_images, return_features=True)
                task_loss = loss_fn(source_outputs, source_labels) / grad_accum_steps
                # Domain discrimination
                _, target_features = seg_model(target_images, return_features=True)
                # Gradient reversal
                source_features_rev = grad_reverse(source_features, lambda_domain)
                target_features_rev = grad_reverse(target_features, lambda_domain)
                # Domain predictions
                source_domain_pred = disc_model(source_features_rev)
                target_domain_pred = disc_model(target_features_rev)
                # Domain labels
                source_domain_labels = torch.zeros(source_domain_pred.size(0), dtype=torch.long, device=device)
                target_domain_labels = torch.ones(target_domain_pred.size(0), dtype=torch.long, device=device)
                domain_preds = torch.cat([source_domain_pred, target_domain_pred], dim=0)
                domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
                domain_loss = nn.CrossEntropyLoss()(domain_preds, domain_labels) / grad_accum_steps
                # Total loss
                total_loss = task_loss + lambda_domain * domain_loss
            scaler.scale(total_loss).backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                scaler.step(task_optimizer)
                scaler.step(domain_optimizer)
                scaler.update()
        else:
            # Segmentation (task) loss
            source_outputs, source_features = seg_model(source_images, return_features=True)
            task_loss = loss_fn(source_outputs, source_labels) / grad_accum_steps
            # Domain discrimination
            _, target_features = seg_model(target_images, return_features=True)
            # Gradient reversal
            source_features_rev = grad_reverse(source_features, lambda_domain)
            target_features_rev = grad_reverse(target_features, lambda_domain)
            # Domain predictions
            source_domain_pred = disc_model(source_features_rev)
            target_domain_pred = disc_model(target_features_rev)
            # Domain labels
            source_domain_labels = torch.zeros(source_domain_pred.size(0), dtype=torch.long, device=device)
            target_domain_labels = torch.ones(target_domain_pred.size(0), dtype=torch.long, device=device)
            domain_preds = torch.cat([source_domain_pred, target_domain_pred], dim=0)
            domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
            domain_loss = nn.CrossEntropyLoss()(domain_preds, domain_labels) / grad_accum_steps
            # Total loss
            total_loss = task_loss + lambda_domain * domain_loss
            total_loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                task_optimizer.step()
                domain_optimizer.step()
        # Metrics (on source)
        with torch.no_grad():
            iou = calculate_iou(source_outputs, source_labels)
            dice = calculate_dice(source_outputs, source_labels)
            acc = calculate_accuracy(source_outputs, source_labels)
        running_task_loss += task_loss.item() * grad_accum_steps
        running_domain_loss += domain_loss.item() * grad_accum_steps
        total_dice += dice.item()
        total_iou += iou.item()
        total_acc += acc.item()
        progress_bar.set_postfix(task_loss=task_loss.item() * grad_accum_steps, domain_loss=domain_loss.item() * grad_accum_steps, total_loss=total_loss.item() * grad_accum_steps, dice=dice.item(), iou=iou.item(), acc=acc.item())
    return running_task_loss/num_batches, running_domain_loss/num_batches, total_dice/num_batches, total_iou/num_batches, total_acc/num_batches

# ---- Validation (source domain only) ----
def evaluate(model, loader, device, epoch, args, loss_fn):
    model.eval()
    running_loss, total_iou, total_dice, total_acc = 0, 0, 0, 0
    num_batches = len(loader)
    progress_bar = tqdm(loader, desc=f"🔵 Validation Epoch {epoch+1}/{args.epochs}", leave=False)
    log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    with torch.no_grad():
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images, return_features=False)
            loss = loss_fn(outputs, labels)
            iou = calculate_iou(outputs, labels)
            dice = calculate_dice(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            running_loss += loss.item()
            total_iou += iou.item()
            total_dice += dice.item()
            total_acc += acc.item()
            progress_bar.set_postfix(val_loss=loss.item(), val_iou=iou.item(), val_dice=dice.item(), val_acc=acc.item())
            if i % 10 == 0:
                log_gpu_usage(os.path.join(args.experiment_dir, args.experiment_name, 'logs', 'gpu_usage.log'))
    return (running_loss / num_batches, total_iou / num_batches, total_dice / num_batches, total_acc / num_batches)

# ---- Main ----
def main(args):
    # Set seed
    torch.manual_seed(args.seed or 42)
    np.random.seed(args.seed or 42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = (args.mixed_precision == 'fp16' and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    print("\n=== GPU Information ===")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU device: {torch.cuda.current_device()}")
    print(f"GPU device name: {torch.cuda.get_device_name()}")
    print(f"Process ID: {os.getpid()}")
    print("=====================\n")
    args.experiment_name = create_experiment_name(args)
    experiment_dir = os.path.join(args.experiment_dir, args.experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    # Save config
    config_file = os.path.join(experiment_dir, 'config.txt')
    with open(config_file, 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    gpu_log_file = os.path.join(log_dir, 'gpu_usage.log')
    log_gpu_usage(gpu_log_file)
    # Data
    # Handle "all" modalities case
    source_modalities = None if args.source_modality.lower() == 'all' else [args.source_modality]
    target_modalities = None if args.target_modality.lower() == 'all' else [args.target_modality]

    # Create datasets from train, dann_add_labeled, dann_add_unlabeled, and target directories
    train_source_main = CombinedDataset(os.path.join(args.data_root, 'train'), modalities=source_modalities, transform=None)
    dann_add_labeled = CombinedDataset(os.path.join(args.data_root, 'dann_add_labeled'), modalities=target_modalities, transform=None)
    val_source = CombinedDataset(os.path.join(args.data_root, 'val'), modalities=target_modalities, transform=None)
    train_target_main = CombinedDataset(os.path.join(args.data_root, 'target'), modalities=target_modalities, transform=None)
    dann_add_unlabeled = CombinedDataset(os.path.join(args.data_root, 'dann_add_unlabeled'), modalities=target_modalities, transform=None)

    # Select n_add_source samples from dann_add_labeled and dann_add_unlabeled if specified
    if args.n_add_source is not None and args.n_add_source < len(dann_add_labeled):
        rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()
        indices = rng.choice(len(dann_add_labeled), args.n_add_source, replace=False)
        dann_add_labeled = Subset(dann_add_labeled, indices)
    if args.n_add_source is not None and args.n_add_source < len(dann_add_unlabeled):
        rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()
        indices = rng.choice(len(dann_add_unlabeled), args.n_add_source, replace=False)
        dann_add_unlabeled = Subset(dann_add_unlabeled, indices)

    # Combine train_source_main and dann_add_labeled for the source set
    train_source = ConcatDataset([train_source_main, dann_add_labeled])
    # Combine train_target_main and dann_add_unlabeled for the target set
    train_target = ConcatDataset([train_target_main, dann_add_unlabeled])

    # Apply n_samples ablation if specified (to both train_source and train_target)
    if args.n_samples is not None:
        rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()
        indices = rng.choice(len(train_source), min(args.n_samples, len(train_source)), replace=False)
        train_source = Subset(train_source, indices)
        indices = rng.choice(len(train_target), min(args.n_samples, len(train_target)), replace=False)
        train_target = Subset(train_target, indices)

    # Print dataset info
    print(f"[INFO] Source (train): {len(train_source_main)} from train/ + {len(dann_add_labeled)} from dann_add_labeled/ = {len(train_source)} total")
    print(f"[INFO] Target (train): {len(train_target_main)} from target/ + {len(dann_add_unlabeled)} from dann_add_unlabeled/ = {len(train_target)} total")
    print(f"[INFO] Validation: {len(val_source)} from val/")

    # Create DataLoaders
    source_loader = DataLoader(train_source, batch_size=args.batch_size, shuffle=True, num_workers=2)
    target_loader = DataLoader(train_target, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_source, batch_size=1, shuffle=False, num_workers=2)

    # Model
    seg_model = UNet3D(in_channels=1, out_channels=4, dropout_rate=args.dropout_rate).to(device)
    # Load pretrained weights if provided
    if getattr(args, 'pretrained_model', None):
        print(f"[INFO] Loading pretrained weights from {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        if 'model_state_dict' in checkpoint:
            seg_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            seg_model.load_state_dict(checkpoint, strict=False)
    # Get actual feature dimension from model using a dynamic shape from the dataset
    with torch.no_grad():
        sample_batch = next(iter(source_loader))[0]  # (B, C, D, H, W) or (B, C, H, W, D)
        dummy_input = sample_batch[:1].to(device)  # Take one sample, keep batch dim
        _, bottleneck_gap = seg_model(dummy_input, return_features=True)
        bottleneck_dim = bottleneck_gap.shape[1]  # Get channel dimension
    disc_model = DomainDiscriminator(bottleneck_dim, hidden_dim=128).to(device)
    # Optimizers
    optim_seg = optim.AdamW(seg_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optim_disc = optim.AdamW(disc_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_seg, mode='max', patience=10, factor=0.1, min_lr=1e-6, verbose=True)
    # Logging
    log_file = os.path.join(log_dir, 'train_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'time', 'train_loss', 'task_loss', 'domain_loss', 'val_loss', 'train_dice', 'val_dice', 'train_iou', 'val_iou', 'train_acc', 'val_acc', 'encoder_frozen'])
    # Training loop
    best_val_dice = 0
    start_time = time.time()
    encoder_frozen = False
    patience_counter = 0
    early_stop = False
    loss_fn = get_loss_fn(args.loss)
    for epoch in range(args.epochs):
        if early_stop:
            break
        epoch_start_time = time.time()
        # Encoder freezing logic (if needed)
        if args.freeze_encoder_epoch is not None:
            if epoch == args.freeze_encoder_epoch and not encoder_frozen:
                print(f"[INFO] 🔒 Freezing encoder at epoch {epoch+1}")
                freeze_encoder(seg_model)
                optim_seg = update_optimizer_for_frozen_encoder(seg_model, optim_seg, args.lr)
                encoder_frozen = True
            elif epoch == args.freeze_encoder_epoch + 1 and encoder_frozen:
                print(f"[INFO] �� Unfreezing encoder at epoch {epoch+1}")
                unfreeze_encoder(seg_model)
                optim_seg = optim.AdamW(seg_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                encoder_frozen = False
        # Training
        train_loss, domain_loss, train_dice, train_iou, train_acc = train_one_epoch_dann(
            seg_model, disc_model, (source_loader, target_loader), optim_seg, optim_disc, device, epoch, args, loss_fn, args.lambda_domain, scaler=scaler, use_fp16=use_fp16)
        # Validation (source domain)
        val_loss, val_iou, val_dice, val_acc = evaluate(seg_model, val_loader, device, epoch, args, loss_fn)
        # scheduler.step(val_dice)
        # Log learning rate
        current_lr = optim_seg.param_groups[0]['lr']
        print(f"[LR] Learning rate after epoch {epoch+1}: {current_lr}")
        epoch_time = time.time() - epoch_start_time
        # Log metrics
        print(f"[EPOCH] 📊 Epoch {epoch+1}/{args.epochs} - Time: {format_time(epoch_time)} | "
              f"Train Loss: {(train_loss + args.lambda_domain * domain_loss):.4f} | Task Loss: {train_loss:.4f} | Domain Loss: {domain_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
              f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Encoder: {'🔒' if encoder_frozen else '🔓'}")
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, epoch_time, train_loss + args.lambda_domain * domain_loss, train_loss, domain_loss, val_loss, train_dice, val_dice, train_iou, val_iou, train_acc, val_acc, encoder_frozen])
        log_gpu_usage(gpu_log_file)
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint_name = f"checkpoint_epoch{epoch+1}_{args.experiment_name}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save({
                'epoch': epoch + 1,
                'train_loss': train_loss + args.lambda_domain * domain_loss,
                'task_loss': train_loss,
                'domain_loss': domain_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
                'encoder_frozen': encoder_frozen,
                'model_state_dict': seg_model.state_dict(),
                'optimizer_state_dict': optim_seg.state_dict(),
            }, checkpoint_path)
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            best_model_path = os.path.join(checkpoint_dir, f"best_model_{args.experiment_name}.pth")
            torch.save({
                'epoch': epoch + 1,
                'train_loss': train_loss + args.lambda_domain * domain_loss,
                'task_loss': train_loss,
                'domain_loss': domain_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
                'encoder_frozen': encoder_frozen,
                'model_state_dict': seg_model.state_dict(),
                'optimizer_state_dict': optim_seg.state_dict(),
            }, best_model_path)
        else:
            if args.early_stopping:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[EARLY STOPPING] No improvement in validation Dice for {args.patience} epochs. Stopping training early at epoch {epoch+1}.")
                    early_stop = True
    # Plot training metrics
    plot_training_metrics(log_file, plots_dir)
    total_time = time.time() - start_time
    print(f"\n[END] ✅ Training completed in {format_time(total_time)}")
    print(f"Best validation Dice score: {best_val_dice:.4f}")
    log_gpu_usage(gpu_log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DANN Training for Multimodal Segmentation')
    parser.add_argument('--data_root', type=str, default='datasets/resampled', help='Root directory of dataset splits')
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Directory to save experiments')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--loss', type=str, default='ce_tversky', choices=['combined', 'ce', 'dice', 'tversky', 'ce_tversky'], help='Loss function to use for training')
    parser.add_argument('--source_modality', type=str, required=True, help='Source modality for DANN experiments')
    parser.add_argument('--target_modality', type=str, required=True, help='Target modality for DANN experiments')
    parser.add_argument('--lambda_domain', type=float, default=0.1, help='Weight for domain loss in DANN experiments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for regularization (default: 0.1)')
    parser.add_argument('--freeze_encoder_epoch', type=int, default=None, help='Epoch to freeze the encoder')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping based on validation Dice')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait for improvement before stopping (used if early stopping is enabled)')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to use for ablation study')
    parser.add_argument('--n_add_source', type=int, default=None, help='Number of additional source volumes from dann_add_labeled/ and dann_add_unlabeled/')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model checkpoint for seg_model')
    args = parser.parse_args()
    main(args)
