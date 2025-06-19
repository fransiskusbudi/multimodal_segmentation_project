import numpy as np

# Load prediction and ground truth
pred = np.load('test_results/test_results_unet_20250616_165825/predictions/pred_10.npy')  # Shape: (192, 192, 192)
gt = np.load('test_results/test_results_unet_20250616_165825/predictions/gt_10.npy')      # Shape: (192, 192, 192)

# Loop through all slices and print unique values
for slice_idx in range(pred.shape[0]):
    pred_slice = pred[slice_idx]
    gt_slice = gt[slice_idx]
    
    print(f"Slice {slice_idx}:")
    print(f"  Prediction unique values: {np.unique(pred_slice)}")
    print(f"  Ground truth unique values: {np.unique(gt_slice)}")
    print("-" * 40)
