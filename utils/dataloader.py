# utils/dataloader.py

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random
import scipy.ndimage
from accelerate import Accelerator

class CombinedDataset(Dataset):
    """
    A combined dataset loader that aggregates all images and labels 
    from datasets stored in train/val/test directories.
    """

    def __init__(self, split_dir, transform=None):
        """
        split_dir: path to the split directory, e.g.,
            '/Users/fransiskusbudi/UoE/Dissertation/multimodal_segmentation_project/datasets/resampled/train'
        transform: optional augmentation function
        """
        self.samples = []
        self.transform = transform
        self.accelerator = Accelerator()

        # Loop through each dataset folder inside the split_dir
        dataset_names = os.listdir(split_dir)
        for dataset_name in dataset_names:
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
            print(f"Loaded {len(self.samples)} samples from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = nib.load(sample['image_path']).get_fdata().astype(np.float32)
        label = nib.load(sample['label_path']).get_fdata().astype(np.int64)

        # Normalize image intensities to [0, 1]
        # image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / (std + 1e-8)
        image = np.clip(image, -5, 5)        # Limit outliers
        image = (image + 5) / 10             # Scale to [0, 1]

        # Add channel dimension (C, H, W, D)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Apply transform if provided
        if self.transform:
            image, label = self.transform(image, label)

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).float()
        label_tensor = (label_tensor > 0.5).float()

        return image_tensor, label_tensor

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

def combined_transform(image, label):
    image, label = random_flip(image, label)
    image, label = random_rotate(image, label)
    return image, label
