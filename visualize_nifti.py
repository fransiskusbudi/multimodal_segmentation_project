import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import os

def visualize_nifti(pred_path, gt_path, image_path, output_dir=None):
    """
    Visualize NIfTI files with interactive slice navigation
    Shows prediction and ground truth overlaid on original scan
    """
    # Load NIfTI files
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)
    image_img = nib.load(image_path)
    pred_data = pred_img.get_fdata()
    gt_data = gt_img.get_fdata()
    image_data = image_img.get_fdata()
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(bottom=0.2)  # Make room for slider
    
    # Initial slice
    slice_idx = pred_data.shape[2] // 2
    
    # Create RGB overlays
    def create_overlay(image_slice, label_slice):
        # First show the original image in grayscale
        overlay = np.zeros((*image_slice.shape, 3))
        overlay[..., 0] = image_slice  # R channel
        overlay[..., 1] = image_slice  # G channel
        overlay[..., 2] = image_slice  # B channel
        
        # Normalize the image to [0, 1]
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
        
        # Overlay the segmentation labels
        # Spleen (label 1) - Red (255, 0, 0)
        mask = label_slice == 1
        overlay[mask] = np.array([1, 0, 0])  # Red
        
        # Liver (label 2) - Orange (255, 165, 0)
        mask = label_slice == 2
        overlay[mask] = np.array([1, 0.65, 0])  # Orange
        
        # Kidneys (label 3) - Green (0, 128, 0)
        mask = label_slice == 3
        overlay[mask] = np.array([0, 0.5, 0])  # Green
        
        return overlay
    
    # Initial visualization
    pred_overlay = create_overlay(image_data[:, :, slice_idx], pred_data[:, :, slice_idx])
    gt_overlay = create_overlay(image_data[:, :, slice_idx], gt_data[:, :, slice_idx])
    
    im1 = ax1.imshow(pred_overlay)
    im2 = ax2.imshow(gt_overlay)
    
    ax1.set_title('Prediction')
    ax2.set_title('Ground Truth')
    fig.suptitle(f'Slice {slice_idx}', y=0.95)
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Slice',
        valmin=0,
        valmax=pred_data.shape[2]-1,
        valinit=slice_idx,
        valstep=1
    )
    
    def update(val):
        slice_idx = int(slider.val)
        # Update RGB overlays
        pred_overlay = create_overlay(image_data[:, :, slice_idx], pred_data[:, :, slice_idx])
        gt_overlay = create_overlay(image_data[:, :, slice_idx], gt_data[:, :, slice_idx])
        
        im1.set_data(pred_overlay)
        im2.set_data(gt_overlay)
        fig.suptitle(f'Slice {slice_idx}', y=0.95)
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add keyboard navigation
    def on_key(event):
        if event.key == 'right':
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == 'left':
            slider.set_val(max(slider.val - 1, slider.valmin))
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Spleen'),
        Patch(facecolor='orange', label='Liver'),
        Patch(facecolor='green', label='Kidneys')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Show plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize NIfTI files with interactive slice navigation')
    parser.add_argument('pred_path', type=str, help='Path to the prediction NIfTI file')
    parser.add_argument('gt_path', type=str, help='Path to the ground truth NIfTI file')
    parser.add_argument('image_path', type=str, help='Path to the original image NIfTI file')
    parser.add_argument('--output_dir', type=str, help='Directory to save visualizations (optional)')
    args = parser.parse_args()
    
    visualize_nifti(args.pred_path, args.gt_path, args.image_path, args.output_dir)

if __name__ == '__main__':
    main() 