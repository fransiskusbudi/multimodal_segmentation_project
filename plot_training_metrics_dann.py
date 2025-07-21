import os
import csv
import matplotlib.pyplot as plt
import sys

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_training_metrics_dann.py <log_file.csv> <save_dir>")
        sys.exit(1)
    log_file = sys.argv[1]
    save_dir = sys.argv[2]
    os.makedirs(save_dir, exist_ok=True)
    plot_training_metrics(log_file, save_dir) 