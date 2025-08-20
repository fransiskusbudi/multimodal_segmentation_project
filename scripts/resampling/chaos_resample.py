import numpy as np
import os
import nibabel as nib
from scipy.ndimage import zoom
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff

# Define paths
input_dir = "/Users/fransiskusbudi/UoE/Dissertation/multimodal_segmentation_project/datasets/chaos/NIfTI_new/MR/images/"
output_dir = "/Users/fransiskusbudi/UoE/Dissertation/multimodal_segmentation_project/datasets/resampled/train/chaos_ras/images/"

labels_dir = "/Users/fransiskusbudi/UoE/Dissertation/multimodal_segmentation_project/datasets/chaos/NIfTI_new/MR/labels/"
labels_out_dir = "/Users/fransiskusbudi/UoE/Dissertation/multimodal_segmentation_project/datasets/resampled/train/chaos_ras/labels/"

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(labels_out_dir, exist_ok=True)

# Target voxel spacing and shape
target_spacing = [1.0, 1.0, 1.0]  # mm
target_shape = [192, 192, 192]

def reorient_to_ras(nii_img):
    orig_ornt = io_orientation(nii_img.affine)
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(orig_ornt, ras_ornt)
    data = nii_img.get_fdata()
    reoriented_data = apply_orientation(data, transform)
    new_affine = nii_img.affine @ inv_ornt_aff(transform, nii_img.shape)
    return nib.Nifti1Image(reoriented_data, new_affine, nii_img.header)

# Process each image
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        print(f"Processing: {filename}")

        # Load and reorient image to RAS
        img_path = os.path.join(input_dir, filename)
        img_nii = nib.load(img_path)
        img_nii = reorient_to_ras(img_nii)
        img_data = img_nii.get_fdata()
        img_affine = img_nii.affine

        # Calculate voxel spacing from affine
        voxel_spacing = np.sqrt((img_affine[:3, :3] ** 2).sum(axis=0))
        print(f"  Original shape: {img_data.shape}")
        print(f"  Original voxel spacing: {voxel_spacing}")

        # Compute scale factors
        scale_factors = voxel_spacing / target_spacing
        print(f"  Scale factors: {scale_factors}")

        # Resample image to isotropic spacing
        img_resampled = zoom(img_data, scale_factors, order=3, mode='nearest', prefilter=False)
        print(f"  Resampled shape: {img_resampled.shape}")

        # Compute resize factors to target shape
        resize_factors = [
            target_shape[i] / img_resampled.shape[i] for i in range(3)
        ]
        print(f"  Resize factors: {resize_factors}")

        # Resize to target shape
        img_resized = zoom(img_resampled, resize_factors, order=3, mode='nearest', prefilter=False)
        print(f"  Final image shape: {img_resized.shape}")

        # Save resampled image
        new_affine = np.copy(img_affine)
        new_affine[:3, :3] = np.diag(target_spacing)
        out_img = nib.Nifti1Image(img_resized.astype(np.float32), new_affine)
        out_path = os.path.join(output_dir, filename)
        nib.save(out_img, out_path)
        print(f"  Saved image to: {out_path}")

        # Process corresponding label
        label_path = os.path.join(labels_dir, filename)
        if os.path.exists(label_path):
            print(f"  Processing label: {filename}")
            label_nii = nib.load(label_path)
            label_nii = reorient_to_ras(label_nii)
            label_data = label_nii.get_fdata()

            # Resample label with nearest neighbor interpolation
            label_resampled = zoom(label_data, scale_factors, order=0, mode='nearest', prefilter=False)
            print(f"  Label resampled shape: {label_resampled.shape}")

            # Resize label to target shape
            label_resized = zoom(label_resampled, resize_factors, order=0, mode='nearest', prefilter=False)
            print(f"  Final label shape: {label_resized.shape}")

            # Save resampled label
            label_out_img = nib.Nifti1Image(label_resized.astype(np.uint8), new_affine)
            label_out_path = os.path.join(labels_out_dir, filename)
            nib.save(label_out_img, label_out_path)
            print(f"  Saved label to: {label_out_path}")
        else:
            print(f"  ⚠️ Label file not found for {filename} — skipping label processing.")

        print()

print("✅ All images and labels have been processed.")
