# utils/dataloader.py

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random
import scipy.ndimage
from accelerate import Accelerator
from monai.transforms import (
    Compose, ScaleIntensityd, RandAffined, Rand3DElasticd,
    RandBiasFieldd, RandGaussianNoised, RandAdjustContrastd,
    RandHistogramShiftd, RandCoarseDropoutd, EnsureTyped
)

class CombinedDataset(Dataset):
    """
    A combined dataset loader that aggregates images and labels 
    from datasets stored in train/val/test directories.
    Supports modality-specific loading (CT only, MRI only, or both).
    """

    def __init__(self, split_dir, transform=None, modalities=None):
        """
        split_dir: path to the split directory, e.g.,
            '/Users/fransiskusbudi/UoE/Dissertation/multimodal_segmentation_project/datasets/resampled/train'
        transform: optional augmentation function
        modalities: list of modalities to include, e.g., ['ct', 'mri'] or ['ct'] or ['mri']
                   if None, includes all modalities
        """
        self.samples = []
        self.transform = transform
        self.accelerator = Accelerator()
        
        # Convert modalities to lowercase for case-insensitive matching
        if modalities is not None:
            self.modalities = [mod.lower() for mod in modalities]
        else:
            self.modalities = None  # Include all modalities
        
        # AMOS dataset mapping for liver, kidneys, and spleen
        self.amos_mapping = {
            0: 0,  # background
            1: 1,  # spleen
            2: 3,  # right kidney -> kidneys class
            3: 3,  # left kidney -> kidneys class
            6: 2,  # liver
        }
        
        # CHAOS dataset mapping
        self.chaos_mapping = {
            0: 0,      # background
            63: 2,     # liver
            126: 3,    # right kidney -> kidneys class
            189: 3,    # left kidney -> kidneys class
            252: 1,    # spleen
        }

        # Loop through each dataset folder inside the split_dir
        dataset_names = os.listdir(split_dir)
        for dataset_name in dataset_names:
            # Check if this dataset matches the requested modalities
            if self.modalities is not None:
                dataset_modality = self._get_modality_from_dataset_name(dataset_name)
                if dataset_modality not in self.modalities:
                    if self.accelerator.is_main_process:
                        print(f"Skipping dataset {dataset_name}: modality '{dataset_modality}' not in requested modalities {self.modalities}")
                    continue
            
            images_dir = os.path.join(split_dir, dataset_name, 'images')
            labels_dir = os.path.join(split_dir, dataset_name, 'labels')

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                if self.accelerator.is_main_process:
                    print(f"Skipping dataset {dataset_name}: missing images or labels directory.")
                continue

            image_files = sorted(os.listdir(images_dir))
            label_files = sorted(os.listdir(labels_dir))

            assert len(image_files) == len(label_files), f"Mismatch between images and labels in {dataset_name}!"

            for img_file, lbl_file in zip(image_files, label_files):
                img_path = os.path.join(images_dir, img_file)
                lbl_path = os.path.join(labels_dir, lbl_file)
                self.samples.append({
                    'image_path': img_path,
                    'label_path': lbl_path,
                    'dataset_name': dataset_name
                })

        if self.accelerator.is_main_process:
            modality_str = f"modalities {self.modalities}" if self.modalities else "all modalities"
            print(f"Loaded {len(self.samples)} samples from {split_dir} ({modality_str})")

    def _get_modality_from_dataset_name(self, dataset_name):
        """
        Extract modality from dataset name.
        Returns 'ct' or 'mri' based on the dataset name.
        """
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower.endswith('_ct'):
            return 'ct'
        elif dataset_name_lower.endswith('_mri'):
            return 'mri'
        else:
            # Default to MRI for unknown datasets
            return 'mri'

    def preprocess_ct(self, image):
        """Preprocess CT image with appropriate window settings."""
        # Typical abdominal window: -160 to 240 HU
        window_min, window_max = -160, 240
        image = np.clip(image, window_min, window_max)
        image = (image - window_min) / (window_max - window_min)
        return image

    # def preprocess_mri(self, image):
    #     """Preprocess MRI image with z-score normalization."""
    #     mean = np.mean(image)
    #     std = np.std(image)
    #     image = (image - mean) / (std + 1e-8)
    #     image = np.clip(image, -5, 5)
    #     image = (image + 5) / 10
    #     return image

    def preprocess_mri(self, image):
        """
        Preprocess MRI image with z-score normalization, percentile-based clipping (1st to 99th), and min-max scaling to [0, 1].
        """
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / (std + 1e-8)

        # Robust clipping to remove outliers
        low, high = np.percentile(image, [1, 99])
        image = np.clip(image, low, high)

        # Min-max scaling to [0, 1]
        image = (image - low) / (high - low + 1e-8)

        return image.astype(np.float32)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = nib.load(sample['image_path']).get_fdata().astype(np.float32)
        label = nib.load(sample['label_path']).get_fdata().astype(np.int64)
        dataset_name = sample['dataset_name']

        # Modality-specific preprocessing
        if dataset_name.lower().endswith('_ct'):
            image = self.preprocess_ct(image)
        elif dataset_name.lower().endswith('_mri'):
            image = self.preprocess_mri(image)
        else:
            image = self.preprocess_mri(image)

        # Handle different dataset label formats
        if dataset_name.startswith('amos'):
            new_label = np.zeros_like(label)
            for old_label, new_label_idx in self.amos_mapping.items():
                new_label[label == old_label] = new_label_idx
            label = new_label
        elif dataset_name.startswith('chaos'):
            new_label = np.zeros_like(label)
            for old_val, new_val in self.chaos_mapping.items():
                if old_val == 63:  # Liver
                    mask = (label >= 55) & (label <= 70)
                elif old_val == 126:  # Right kidney
                    mask = (label >= 110) & (label <= 135)
                elif old_val == 189:  # Left kidney
                    mask = (label >= 175) & (label <= 200)
                elif old_val == 252:  # Spleen
                    mask = (label >= 240) & (label <= 255)
                else:  # Background
                    mask = (label == 0)
                new_label[mask] = new_val
            label = new_label
        elif dataset_name.startswith('ts'):
            pass  # labels are already correct
        elif dataset_name == 'btcv':
            pass

        # Add channel dimension (C, H, W, D)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Prepare sample for MONAI transform
        sample_dict = {'image': image, 'label': label}
        if self.transform:
            transformed = self.transform(sample_dict)
            image = transformed['image']
            label = transformed['label']

        image_tensor = torch.as_tensor(image).float()
        label_tensor = torch.as_tensor(label).long()
        return image_tensor, label_tensor

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

# Example transforms
def random_flip(image, label):
    axes = [1, 2, 3]
    for axis in axes:
        if random.random() > 0.5:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    return image.copy(), label.copy()

def random_rotate(image, label, max_angle=15):
    axes = [(1, 2), (1, 3), (2, 3)]
    angle = random.uniform(-max_angle, max_angle)
    axis = random.choice(axes)
    image = scipy.ndimage.rotate(image, angle, axes=axis, reshape=False, order=1, mode='nearest')
    label = scipy.ndimage.rotate(label, angle, axes=axis, reshape=False, order=0, mode='nearest')
    return image.copy(), label.copy()

def combined_transform():
    return Compose([
        # ScaleIntensityd(keys=["image"]),
        # EnsureTyped(keys=["image", "label"]),
        # RandAffined(
        #     keys=["image"],
        #     rotate_range=(0.1, 0.1, 0.1),
        #     scale_range=(0.1, 0.1, 0.1),
        #     prob=1,
        #     padding_mode="border"
        # ),
        # RandAffined(
        #     keys=["label"],
        #     rotate_range=(0.1, 0.1, 0.1),
        #     scale_range=(0.1, 0.1, 0.1),
        #     prob=1,
        #     padding_mode="zeros"
        # ),
        # Rand3DElasticd(
        #     keys=["image", "label"],
        #     sigma_range=(5, 8),
        #     magnitude_range=(100, 200),
        #     spatial_size=(96, 96, 96),
        #     prob=1,
        #     mode=("bilinear", "nearest")
        # ),
        RandBiasFieldd(keys=["image"], prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
        RandHistogramShiftd(keys=["image"], prob=0.3, num_control_points=5),
        RandCoarseDropoutd(
            keys=["image", "label"],
            holes=2,
            spatial_size=(16, 16, 16),
            fill_value=0.0,
            prob=0.3
        )
    ])

if __name__ == "__main__":
    # Example usage: visualize a transformed sample
    import sys
    import matplotlib.pyplot as plt

    # Set these paths as needed for your environment
    example_data_root = "/home/s2670828/multimodal_segmentation_project/datasets/resampled/train"  # Change if needed
    modalities = ['mri']  # or ['ct'], ['mri']

    # Create dataset with MONAI transform
    dataset = CombinedDataset(example_data_root, transform=combined_transform(), modalities=modalities)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Get a sample
    img, lbl = dataset[0]
    # Also load the original (untransformed) image and label for comparison
    orig_dataset = CombinedDataset(example_data_root, transform=None, modalities=modalities)
    orig_img, orig_lbl = orig_dataset[0]

    # Use the middle slice index from the original image for both visualizations
    orig_slice_idx = orig_img.shape[-1] // 2

    # Handle possible shape change after transformation
    def get_slice(tensor, idx):
        # If shape changed, try to map the original index proportionally
        orig_depth = orig_img.shape[-1]
        new_depth = tensor.shape[-1]
        mapped_idx = int(idx * new_depth / orig_depth)
        mapped_idx = min(max(mapped_idx, 0), new_depth - 1)
        return tensor[0, :, :, mapped_idx].cpu().numpy()

    orig_slice_img = get_slice(orig_img, orig_slice_idx)
    orig_slice_lbl = get_slice(orig_lbl, orig_slice_idx)
    slice_img = get_slice(img, orig_slice_idx)
    slice_lbl = get_slice(lbl, orig_slice_idx)

    # Show side-by-side comparison and save as one PNG
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].imshow(orig_slice_img, cmap='gray')
    axs[0, 0].set_title('Original Image (mid slice)')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(slice_img, cmap='gray')
    axs[0, 1].set_title('Transformed Image (same slice)')
    axs[0, 1].axis('off')
    axs[1, 0].imshow(orig_slice_lbl, cmap='tab10', vmin=0, vmax=3)
    axs[1, 0].set_title('Original Label (mid slice)')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(slice_lbl, cmap='tab10', vmin=0, vmax=3)
    axs[1, 1].set_title('Transformed Label (same slice)')
    axs[1, 1].axis('off')
    plt.tight_layout()
    save_dir = "/home/s2670828/multimodal_segmentation_project/experiments/transformed_example_output"
    comparison_path = os.path.join(save_dir, "comparison.png")
    plt.savefig(comparison_path)
    plt.show()
    print(f"Also saved side-by-side comparison as {comparison_path}")
