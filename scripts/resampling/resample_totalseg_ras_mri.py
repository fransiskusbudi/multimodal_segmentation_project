import numpy as np
import os
import nibabel as nib
from scipy.ndimage import zoom
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff

# Define paths
input_dir = "../datasets/TotalsegmentatorMRI_abdomen_only/"
output_dir = "../datasets/resampled/totalseg_mri_ras/images/"
labels_out_dir = "../datasets/resampled/totalseg_mri_ras/labels/"

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

# List all subject directories
subject_dirs = [d for d in os.listdir(input_dir) if d.startswith('s') and os.path.isdir(os.path.join(input_dir, d))]
subject_dirs.sort()
subject_dirs = subject_dirs[:100]  # Only process the first 100 subjects

for subject in subject_dirs:
    print(f"Processing: {subject}")
    subj_path = os.path.join(input_dir, subject)
    mri_path = os.path.join(subj_path, "mri.nii.gz")
    seg_dir = os.path.join(subj_path, "segmentations")
    
    # Load and reorient MRI to RAS
    img_nii = nib.load(mri_path)
    img_nii = reorient_to_ras(img_nii)
    print(nib.aff2axcodes(img_nii.affine))
    img_data = img_nii.get_fdata()
    img_affine = img_nii.affine
    
    # Calculate voxel spacing from affine
    voxel_spacing = np.sqrt((img_affine[:3, :3] ** 2).sum(axis=0))
    print(f"  Original shape: {img_data.shape}")
    print(f"  Original voxel spacing: {voxel_spacing}")
    
    # Compute scale factors
    scale_factors = voxel_spacing / target_spacing
    print(f"  Scale factors: {scale_factors}")
    
    # Resample MRI to isotropic spacing
    img_resampled = zoom(img_data, scale_factors, order=3, mode='nearest', prefilter=False)
    print(f"  Resampled shape: {img_resampled.shape}")
    
    # Compute resize factors to target shape
    resize_factors = [target_shape[i] / img_resampled.shape[i] for i in range(3)]
    print(f"  Resize factors: {resize_factors}")
    
    # Resize to target shape
    img_resized = zoom(img_resampled, resize_factors, order=3, mode='nearest', prefilter=False)
    print(f"  Final image shape: {img_resized.shape}")
    
    # Save resampled MRI
    new_affine = np.copy(img_affine)
    new_affine[:3, :3] = np.diag(target_spacing)
    out_img = nib.Nifti1Image(img_resized.astype(np.float32), new_affine)
    out_path = os.path.join(output_dir, f"{subject}.nii.gz")
    nib.save(out_img, out_path)
    print(f"  Saved image to: {out_path}")
    
    # Process and combine label masks
    label_names = {
        'spleen': 1,
        'liver': 2,
        'kidney_left': 3,
        'kidney_right': 3
    }
    label_combined = np.zeros(target_shape, dtype=np.uint8)
    for label, value in label_names.items():
        label_path = os.path.join(seg_dir, f"{label}.nii.gz")
        if os.path.exists(label_path):
            print(f"  Processing label: {label}")
            label_nii = nib.load(label_path)
            label_nii = reorient_to_ras(label_nii)
            label_data = label_nii.get_fdata()
            # Resample label with nearest neighbor interpolation
            label_resampled = zoom(label_data, scale_factors, order=0, mode='nearest', prefilter=False)
            # Resize label to target shape
            label_resized = zoom(label_resampled, resize_factors, order=0, mode='nearest', prefilter=False)
            # Combine into one label map using the new mapping
            label_combined[label_resized > 0] = value
            print(f"    Added {label} as value {value}")
        else:
            print(f"    ⚠️ Label file not found for {label}")
    # Save combined label
    label_out_img = nib.Nifti1Image(label_combined.astype(np.uint8), new_affine)
    label_out_path = os.path.join(labels_out_dir, f"{subject}.nii.gz")
    nib.save(label_out_img, label_out_path)
    print(f"  Saved combined label to: {label_out_path}")
    print()

print("✅ All subjects have been processed.") 