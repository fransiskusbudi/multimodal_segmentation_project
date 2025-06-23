# Modality Selection Usage Examples

This document shows how to use the modality selection feature in training, testing, and fine-tuning.

## Dataset Structure

Your dataset should be organized as follows:
```
datasets/resampled/
├── train/
│   ├── amos_ct/
│   │   ├── images/
│   │   └── labels/
│   ├── amos_mri/
│   │   ├── images/
│   │   └── labels/
│   ├── chaos_ct/
│   │   ├── images/
│   │   └── labels/
│   └── chaos_mri/
│       ├── images/
│       └── labels/
├── val/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

The dataloader automatically detects modalities based on dataset names ending with `_ct` or `_mri`.

## Training Examples

### Train on all modalities (CT + MRI)
```bash
python train_unet.py \
    --data_root datasets/resampled \
    --modalities all \
    --batch_size 2 \
    --epochs 100 \
    --lr 0.001
```

### Train on CT only
```bash
python train_unet.py \
    --data_root datasets/resampled \
    --modalities ct \
    --batch_size 2 \
    --epochs 100 \
    --lr 0.001
```

### Train on MRI only
```bash
python train_unet.py \
    --data_root datasets/resampled \
    --modalities mri \
    --batch_size 2 \
    --epochs 100 \
    --lr 0.001
```

### Train on specific modalities (CT and MRI)
```bash
python train_unet.py \
    --data_root datasets/resampled \
    --modalities ct,mri \
    --batch_size 2 \
    --epochs 100 \
    --lr 0.001
```

## Testing Examples

### Test on all modalities
```bash
python test_model.py \
    --model_path experiments/your_model/checkpoints/best_model.pth \
    --data_root datasets/resampled \
    --modalities all \
    --experiment_dir test_results \
    --model_name your_model
```

### Test on CT only
```bash
python test_model.py \
    --model_path experiments/your_model/checkpoints/best_model.pth \
    --data_root datasets/resampled \
    --modalities ct \
    --experiment_dir test_results \
    --model_name your_model
```

### Test on MRI only
```bash
python test_model.py \
    --model_path experiments/your_model/checkpoints/best_model.pth \
    --data_root datasets/resampled \
    --modalities mri \
    --experiment_dir test_results \
    --model_name your_model
```

## Fine-tuning Examples

### Fine-tune on CT data
```bash
python finetune_ct.py \
    --pretrained_model experiments/pretrained_model/checkpoints/best_model.pth \
    --data_root datasets/resampled \
    --modalities ct \
    --batch_size 1 \
    --epochs 50 \
    --lr 0.0001
```

### Fine-tune on MRI data
```bash
python finetune_ct.py \
    --pretrained_model experiments/pretrained_model/checkpoints/best_model.pth \
    --data_root datasets/resampled \
    --modalities mri \
    --batch_size 1 \
    --epochs 50 \
    --lr 0.0001
```

### Fine-tune on both CT and MRI
```bash
python finetune_ct.py \
    --pretrained_model experiments/pretrained_model/checkpoints/best_model.pth \
    --data_root datasets/resampled \
    --modalities ct,mri \
    --batch_size 1 \
    --epochs 50 \
    --lr 0.0001
```

## SLURM Script Examples

### Training Script (run_training.sh)
```bash
# Modify the MODALITIES variable in the script
MODALITIES="ct"  # For CT only
MODALITIES="mri"  # For MRI only
MODALITIES="ct,mri"  # For both CT and MRI
MODALITIES="all"  # For all modalities
```

### Testing Script (run_testing.sh)
```bash
# Modify the MODALITIES variable in the script
MODALITIES="ct"  # For CT only
MODALITIES="mri"  # For MRI only
MODALITIES="ct,mri"  # For both CT and MRI
MODALITIES="all"  # For all modalities
```

### Fine-tuning Script (run_finetune_ct.sh)
```bash
# Modify the MODALITIES variable in the script
MODALITIES="ct"  # For CT only
MODALITIES="mri"  # For MRI only
MODALITIES="ct,mri"  # For both CT and MRI
MODALITIES="all"  # For all modalities
```

## Notes

1. **Case Insensitive**: Modality names are case-insensitive. `CT`, `ct`, `Ct` all work the same.
2. **Dataset Naming**: Make sure your dataset folders end with `_ct` or `_mri` for automatic modality detection.
3. **Default Behavior**: If no modality is specified or `all` is used, all available modalities will be included.
4. **Logging**: The dataloader will print which datasets are being loaded and which are being skipped based on modality selection.
5. **Compatibility**: This feature works with all existing functionality including multi-GPU training, checkpointing, and evaluation metrics. 