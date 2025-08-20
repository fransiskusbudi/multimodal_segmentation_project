import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from accelerate import Accelerator
from utils.dataloader import CombinedDataset, combined_transform
from models.unet import UNet3D
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from utils.metrics import calculate_dice as torch_calculate_dice, calculate_iou as torch_calculate_iou
import json
import csv
import time
import random

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def load_model(model_path, device):
    """Load the trained model."""
    model = UNet3D(in_channels=1, out_channels=1)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def save_prediction(pred_np, original_path, output_dir, filename):
    """
    Save prediction as NIfTI file
    """
    # Load original image to get header
    original_img = nib.load(original_path)
    
    # Create new NIfTI image with prediction
    pred_img = nib.Nifti1Image(pred_np, original_img.affine, original_img.header)
    
    # Save prediction
    output_path = os.path.join(output_dir, filename)
    nib.save(pred_img, output_path)
    
    return output_path

def visualize_prediction(image, label, pred, save_path):
    """
    Visualize prediction with original image and ground truth
    Shows all three views: axial, sagittal, and coronal
    Finds the best slices that contain organs for better visualization
    """
    label = np.squeeze(label)
    pred = np.squeeze(pred)
    
    def find_best_slice(label_volume, axis):
        """Find the slice with the most organ pixels along a given axis"""
        if axis == 0:  # sagittal
            organ_pixels_per_slice = np.sum(label_volume > 0, axis=(1, 2))
        elif axis == 1:  # coronal
            organ_pixels_per_slice = np.sum(label_volume > 0, axis=(0, 2))
        else:  # axial
            organ_pixels_per_slice = np.sum(label_volume > 0, axis=(0, 1))
        
        # Find slice with maximum organ pixels
        best_slice = np.argmax(organ_pixels_per_slice)
        
        # If no organs found, fall back to middle slice
        if organ_pixels_per_slice[best_slice] == 0:
            best_slice = label_volume.shape[axis] // 2
        
        return best_slice
    
    # Find best slices that contain organs for each view
    axial_slice = find_best_slice(label, 2)  # z-direction for axial
    sagittal_slice = find_best_slice(label, 0)  # x-direction for sagittal
    coronal_slice = find_best_slice(label, 1)  # y-direction for coronal
    
    # Create figure with 3x3 subplots (3 views x 3 types)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Function to create overlay
    def create_overlay(image_slice, label_slice):
        overlay = np.zeros((*image_slice.shape, 3))
        # First show the original image in grayscale
        overlay[..., 0] = image_slice
        overlay[..., 1] = image_slice
        overlay[..., 2] = image_slice
        # Normalize to [0,1]
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
        
        # Overlay the segmentation labels
        # Spleen (label 1) - Red (255, 0, 0)
        mask = label_slice == 1
        overlay[mask] = np.array([1, 0, 0])
        
        # Liver (label 2) - Orange (255, 165, 0)
        mask = label_slice == 2
        overlay[mask] = np.array([1, 0.65, 0])
        
        # Kidneys (label 3) - Green (0, 128, 0)
        mask = label_slice == 3
        overlay[mask] = np.array([0, 0.5, 0])
        
        return overlay
    
    # Axial view (top row)
    # Original
    axes[0,0].imshow(np.rot90(image[:, :, axial_slice]), cmap='gray')
    axes[0,0].set_title('Axial - Original', pad=20)
    axes[0,0].axis('off')
    
    # Ground truth
    gt_overlay = create_overlay(image[:, :, axial_slice], label[:, :, axial_slice])
    axes[0,1].imshow(np.rot90(gt_overlay))
    axes[0,1].set_title('Axial - Ground Truth', pad=20)
    axes[0,1].axis('off')
    
    # Prediction
    pred_overlay = create_overlay(image[:, :, axial_slice], pred[:, :, axial_slice])
    axes[0,2].imshow(np.rot90(pred_overlay))
    axes[0,2].set_title('Axial - Prediction', pad=20)
    axes[0,2].axis('off')
    
    # Sagittal view (middle row)
    # Original
    axes[1,0].imshow(np.rot90(image[sagittal_slice, :, :]), cmap='gray')
    axes[1,0].set_title('Sagittal - Original', pad=20)
    axes[1,0].axis('off')
    
    # Ground truth
    gt_overlay = create_overlay(image[sagittal_slice, :, :], label[sagittal_slice, :, :])
    axes[1,1].imshow(np.rot90(gt_overlay))
    axes[1,1].set_title('Sagittal - Ground Truth', pad=20)
    axes[1,1].axis('off')
    
    # Prediction
    pred_overlay = create_overlay(image[sagittal_slice, :, :], pred[sagittal_slice, :, :])
    axes[1,2].imshow(np.rot90(pred_overlay))
    axes[1,2].set_title('Sagittal - Prediction', pad=20)
    axes[1,2].axis('off')
    
    # Coronal view (bottom row)
    # Original
    axes[2,0].imshow(np.rot90(image[:, coronal_slice, :]), cmap='gray')
    axes[2,0].set_title('Coronal - Original', pad=20)
    axes[2,0].axis('off')
    
    # Ground truth
    gt_overlay = create_overlay(image[:, coronal_slice, :], label[:, coronal_slice, :])
    axes[2,1].imshow(np.rot90(gt_overlay))
    axes[2,1].set_title('Coronal - Ground Truth', pad=20)
    axes[2,1].axis('off')
    
    # Prediction
    pred_overlay = create_overlay(image[:, coronal_slice, :], pred[:, coronal_slice, :])
    axes[2,2].imshow(np.rot90(pred_overlay))
    axes[2,2].set_title('Coronal - Prediction', pad=20)
    axes[2,2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Spleen'),
        Patch(facecolor='orange', label='Liver'),
        Patch(facecolor='green', label='Kidneys')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=3, bbox_transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def create_test_results_dir(base_dir, model_name):
    """
    Create a unique directory for test results using timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(base_dir, f"test_results_{model_name}_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)
    return test_dir

def test_model(model, test_loader, accelerator, args):
    """
    Test the model and save predictions
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    # Create directories for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.experiment_dir, f'test_results_{args.model_name}_{timestamp}')
    predictions_dir = os.path.join(results_dir, 'predictions')
    metrics_dir = os.path.join(results_dir, 'metrics')
    visualizations_dir = os.path.join(results_dir, 'visualizations')

    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    # Save test configuration
    test_config_file = os.path.join(results_dir, 'test_config.txt')
    with open(test_config_file, 'w') as f:
        f.write(f"Test Configuration:\n")
        f.write(f"Seed: {args.seed}\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # Initialize metrics
    metrics = {
        'dice_spleen': [],
        'dice_liver': [],
        'dice_kidneys': [],
        'iou_spleen': [],
        'iou_liver': [],
        'iou_kidneys': []
    }
    per_sample_metrics = []  # For CSV
    
    with torch.no_grad():
        total_inference_time = 0.0
        for i, (image, label) in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                print(f"\nProcessing sample {i+1}/{len(test_loader)}")
                start_time = time.time()
                outputs = model(image)
                inference_time = time.time() - start_time
                print(f"Inference time: {inference_time:.4f} seconds")
                total_inference_time += inference_time
                
                # outputs: (B, C, ...), label: (B, 1, ...)
                # For per-class metrics, use torch logic as in metrics.py
                pred_classes = torch.argmax(outputs, dim=1)  # (B, ...)
                label_classes = label.squeeze(1)  # (B, ...)
                
                # For each class (1=spleen, 2=liver, 3=kidneys)
                dice_spleen = None
                dice_liver = None
                dice_kidneys = None
                iou_spleen = None
                iou_liver = None
                iou_kidneys = None
                for class_idx, name in zip([1,2,3], ['spleen','liver','kidneys']):
                    pred_mask = (pred_classes == class_idx)
                    label_mask = (label_classes == class_idx)
                    # Only compute if present in label
                    if label_mask.sum() > 0:
                        intersection = (pred_mask & label_mask).sum().float()
                        union = pred_mask.sum() + label_mask.sum()
                        dice = (2. * intersection + 1e-5) / (union + 1e-5)
                        iou = (intersection + 1e-5) / (pred_mask.sum() + label_mask.sum() - intersection + 1e-5)
                    else:
                        dice = torch.tensor(0.0)
                        iou = torch.tensor(0.0)
                    if name == 'spleen':
                        dice_spleen = dice.item()
                        iou_spleen = iou.item()
                    elif name == 'liver':
                        dice_liver = dice.item()
                        iou_liver = iou.item()
                    elif name == 'kidneys':
                        dice_kidneys = dice.item()
                        iou_kidneys = iou.item()
                
                print(f"Metrics - Spleen: Dice={dice_spleen:.4f}, IoU={iou_spleen:.4f}")
                print(f"Metrics - Liver: Dice={dice_liver:.4f}, IoU={iou_liver:.4f}")
                print(f"Metrics - Kidneys: Dice={dice_kidneys:.4f}, IoU={iou_kidneys:.4f}")
                
                metrics['dice_spleen'].append(dice_spleen)
                metrics['dice_liver'].append(dice_liver)
                metrics['dice_kidneys'].append(dice_kidneys)
                metrics['iou_spleen'].append(iou_spleen)
                metrics['iou_liver'].append(iou_liver)
                metrics['iou_kidneys'].append(iou_kidneys)
                
                # Save visualization and NIfTI as before
                pred_np = pred_classes.cpu().numpy()
                label_np = label_classes.cpu().numpy()
                image_np = image.cpu().numpy()
                original_nifti_path = test_loader.dataset.samples[i]['image_path']
                original_filename = os.path.splitext(os.path.basename(original_nifti_path))[0]
                vis_path = os.path.join(visualizations_dir, f'{original_filename}_pred.png')
                visualize_prediction(image_np[0, 0], label_np[0], pred_np[0], vis_path)
                original_nifti = nib.load(original_nifti_path)
                pred_nifti = nib.Nifti1Image(pred_np[0], affine=original_nifti.affine, header=original_nifti.header)
                pred_nifti_path = os.path.join(predictions_dir, f'{original_filename}_pred.nii.gz')
                nib.save(pred_nifti, pred_nifti_path)
                
                per_sample_metrics.append({
                    'filename': original_filename,
                    'dice_spleen': dice_spleen,
                    'dice_liver': dice_liver,
                    'dice_kidneys': dice_kidneys,
                    'iou_spleen': iou_spleen,
                    'iou_liver': iou_liver,
                    'iou_kidneys': iou_kidneys,
                    'inference_time': inference_time
                })
                
                all_predictions.append(pred_np)
                all_labels.append(label_np)
                
                print(f"Successfully processed sample {i+1}")
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save per-sample metrics to CSV
    csv_path = os.path.join(metrics_dir, 'per_sample_metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'dice_spleen', 'dice_liver', 'dice_kidneys', 
                     'iou_spleen', 'iou_liver', 'iou_kidneys', 'inference_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_sample_metrics:
            writer.writerow(row)
    
    # Calculate and save overall metrics
    overall_metrics = {
        'mean_dice_spleen': np.mean(metrics['dice_spleen']),
        'mean_dice_liver': np.mean(metrics['dice_liver']),
        'mean_dice_kidneys': np.mean(metrics['dice_kidneys']),
        'mean_iou_spleen': np.mean(metrics['iou_spleen']),
        'mean_iou_liver': np.mean(metrics['iou_liver']),
        'mean_iou_kidneys': np.mean(metrics['iou_kidneys']),
        # Overall means across all classes
        'mean_dice_overall': np.mean([np.mean(metrics['dice_spleen']), 
                                     np.mean(metrics['dice_liver']), 
                                     np.mean(metrics['dice_kidneys'])]),
        'mean_iou_overall': np.mean([np.mean(metrics['iou_spleen']), 
                                    np.mean(metrics['iou_liver']), 
                                    np.mean(metrics['iou_kidneys'])]),
        'total_inference_time': total_inference_time
    }
    
    with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    
    print(f"\nTest Results saved in: {results_dir}")
    print("\nOverall Metrics:")
    print(f"Spleen - Dice: {overall_metrics['mean_dice_spleen']:.4f}, IoU: {overall_metrics['mean_iou_spleen']:.4f}")
    print(f"Liver - Dice: {overall_metrics['mean_dice_liver']:.4f}, IoU: {overall_metrics['mean_iou_liver']:.4f}")
    print(f"Kidneys - Dice: {overall_metrics['mean_dice_kidneys']:.4f}, IoU: {overall_metrics['mean_iou_kidneys']:.4f}")
    print(f"\nOverall Mean - Dice: {overall_metrics['mean_dice_overall']:.4f}, IoU: {overall_metrics['mean_iou_overall']:.4f}")

def main(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load model
    model = UNet3D(in_channels=1, out_channels=4)  # 4 classes: background + spleen + liver + kidneys
    checkpoint = torch.load(args.model_path, map_location=accelerator.device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # Remove 'module.' prefix if present
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(accelerator.device)
    
    # Prepare test dataset
    test_dir = os.path.join(args.data_root, 'test')
    test_dataset = CombinedDataset(test_dir, transform=None, modalities=args.modalities)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    # Prepare model and data loader with accelerator
    model, test_loader = accelerator.prepare(model, test_loader)
    
    print('test')

    # Run testing
    print(f"\n[TEST] 🧪 Starting Testing with model: {args.model_name}")
    test_model(model, test_loader, accelerator, args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test UNet3D model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--experiment_dir', type=str, required=True, help='Base directory for saving test results')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for result folder')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--modalities', type=str, default='all', help='Comma-separated list of modalities to include')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Process modalities argument
    if args.modalities.lower() == 'all':
        args.modalities = None  # None means include all modalities
    else:
        args.modalities = [mod.strip().lower() for mod in args.modalities.split(',')]
    
    main(args) 