import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from accelerate import Accelerator
from utils.dataloader import CombinedDataset, combined_transform
from models.unet import UNet
import nibabel as nib
from pathlib import Path

def load_model(model_path, device):
    """Load the trained model."""
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_prediction(pred, original_path, save_dir, filename):
    """Save prediction as NIfTI file."""
    # Load original image to get header
    original_img = nib.load(original_path)
    
    # Convert prediction to NIfTI
    pred_nii = nib.Nifti1Image(pred, original_img.affine, original_img.header)
    
    # Save prediction
    save_path = os.path.join(save_dir, f"pred_{filename}")
    nib.save(pred_nii, save_path)
    return save_path

def visualize_prediction(image, label, prediction, slice_idx, save_path=None):
    """Visualize original image, ground truth, and prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(image[slice_idx], cmap='gray')
    axes[1].imshow(label[slice_idx], cmap='jet', alpha=0.4)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(image[slice_idx], cmap='gray')
    axes[2].imshow(prediction[slice_idx], cmap='jet', alpha=0.4)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_model(args):
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Prepare dataset
    test_dataset = CombinedDataset(split_dir=args.test_dir, transform=combined_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Wrap model and data loader with accelerator
    model, test_loader = accelerator.prepare(model, test_loader)
    
    # Testing loop
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            # Get predictions
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            
            # Convert to numpy for visualization
            image_np = images[0, 0].cpu().numpy()
            label_np = labels[0, 0].cpu().numpy()
            pred_np = predictions[0, 0].cpu().numpy()
            
            # Save prediction as NIfTI
            original_path = test_dataset.image_paths[idx]
            pred_path = save_prediction(
                pred_np,
                original_path,
                os.path.join(args.output_dir, 'predictions'),
                os.path.basename(original_path)
            )
            
            # Visualize middle slice
            mid_slice = image_np.shape[0] // 2
            vis_path = os.path.join(
                args.output_dir,
                'visualizations',
                f'vis_{os.path.basename(original_path).replace(".nii.gz", ".png")}'
            )
            visualize_prediction(image_np, label_np, pred_np, mid_slice, vis_path)
            
            if accelerator.is_main_process:
                print(f"Processed {idx+1}/{len(test_loader)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save results')
    args = parser.parse_args()
    
    test_model(args) 