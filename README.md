# Cross-Modality Generalization for Abdominal Organ Segmentation: MRI-to-CT Transfer with Limited Labels

This repository contains the implementation for a dissertation project on multimodal medical image segmentation, focusing on domain adaptation and knowledge distillation techniques for improved performance across different imaging modalities (CT and MRI).

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This project implements and evaluates several approaches for multimodal medical image segmentation:

1. **Baseline U-Net Training**: Standard 3D U-Net training on multimodal data
2. **Fine-tuning Strategies**: Various fine-tuning approaches for limited data scenarios
3. **Knowledge Distillation**: Transfer learning from teacher to student models
4. **Domain Adversarial Neural Networks (DANN)**: Domain adaptation between CT and MRI modalities

### Key Features

- **Multi-class Segmentation**: Liver, kidneys, and spleen segmentation
- **Modality-specific Training**: Support for CT-only, MRI-only, or combined training
- **Domain Adaptation**: DANN implementation for cross-modality generalization
- **Knowledge Distillation**: Teacher-student framework for model compression
- **Comprehensive Evaluation**: Dice, IoU, and accuracy metrics
- **Visualization Tools**: Interactive NIfTI visualization and training plots

## 📁 Project Structure

```
multimodal_segmentation_project/
├── models/
│   ├── unet.py              # Standard 3D U-Net implementation
│   └── unet_dann.py         # U-Net with domain adaptation features
├── utils/
│   ├── dataloader.py        # Multi-modal data loading and augmentation
│   └── metrics.py           # Evaluation metrics (Dice, IoU, accuracy)
├── scripts/
│   ├── plotting/
│   │   ├── plot_results.py              # Results visualization
│   │   ├── plot_results_mri_baseline.py # MRI baseline results
│   │   └── plot_results_line_graph.ipynb # Line graph plotting notebook
│   └── resampling/
│       ├── spacing.ipynb                 # Spacing analysis notebook
│       ├── amos_ct_resample.py          # AMOS CT resampling script
│       ├── chaos_resample.py            # CHAOS resampling script
│       └── resample_totalseg_ras_mri.py # TotalSeg RAS MRI resampling
├── experiments/             # Training outputs and checkpoints
├── datasets/                # Dataset storage (not included in repo)
├── logs/                    # Training logs
├── test_results/            # Test outputs and visualizations
├── notebooks/               # Jupyter notebooks for analysis
│   ├── testing.ipynb        # Testing and evaluation
│   ├── viz_amos.ipynb       # AMOS dataset visualization
│   └── spacing.ipynb        # Spacing analysis
├── train_unet.py           # Baseline U-Net training
├── train_dann.py           # DANN training script
├── distill_unet.py         # Knowledge distillation training
├── finetune_ct.py          # Fine-tuning script
├── test_model.py           # Model evaluation script
├── main.py                 # Main orchestration script
├── visualize_nifti.py      # Interactive NIfTI viewer
├── npy_reader.py           # NumPy file reader utility
├── requirements.txt        # Python dependencies
└── run_*.sh               # SLURM scripts for cluster execution
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) 11 GB VRAM minimum
- SLURM cluster access (for batch jobs)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/multimodal_segmentation_project.git
cd multimodal_segmentation_project
```

2. **Create conda environment**:
```bash
conda create -n diss python=3.8
conda activate diss
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies

- **PyTorch**: Deep learning framework
- **Accelerate**: Distributed training and mixed precision
- **MONAI**: Medical AI framework for data augmentation
- **Nibabel**: NIfTI medical image processing
- **Matplotlib**: Visualization and plotting
- **NumPy/SciPy**: Scientific computing

## 📊 Dataset Preparation

### Dataset Structure

Organize your datasets as follows:

```
datasets/resampled/
├── train/
│   ├── amos_ct/
│   │   ├── images/          # CT images (.nii.gz)
│   │   └── labels/          # Segmentation labels (.nii.gz)
│   ├── amos_mri/
│   │   ├── images/          # MRI images (.nii.gz)
│   │   └── labels/          # Segmentation labels (.nii.gz)
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

### Supported Datasets

- **AMOS**: Abdominal Multi-Organ Segmentation
- **CHAOS**: Combined Healthy Abdominal Organ Segmentation

### Data Preprocessing

1. **Resampling**: Ensure consistent voxel spacing across datasets
   ```bash
   python scripts/resampling/amos_ct_resample.py
   python scripts/resampling/chaos_resample.py
   python scripts/resampling/resample_totalseg_ras_mri.py
   ```
2. **Normalization**: Intensity normalization for each modality
3. **Label Mapping**: Standardize organ labels across datasets:
   - Label 0: Background
   - Label 1: Spleen
   - Label 2: Liver
   - Label 3: Kidneys (both left and right)

## 🎮 Usage

### 1. Baseline Training

#### Train on all modalities:
```bash
sbatch run_training.sh
```

#### Train on specific modality (CT):
```bash
sbatch run_training_ct_1.sh    # 1 sample
sbatch run_training_ct_5.sh    # 5 samples
sbatch run_training_ct_10.sh   # 10 samples
sbatch run_training_ct_25.sh   # 25 samples
sbatch run_training_ct_50.sh   # 50 samples
sbatch run_training_ct_100.sh  # 100 samples
```

### 2. Domain Adversarial Training (DANN)

```bash
sbatch run_dann_n1.sh     # 1 additional source sample
sbatch run_dann_n5.sh     # 5 additional source samples
sbatch run_dann_n10.sh    # 10 additional source samples
sbatch run_dann_n25.sh    # 25 additional source samples
sbatch run_dann_n50.sh    # 50 additional source samples
sbatch run_dann_n100.sh   # 100 additional source samples
```

### 3. Knowledge Distillation

```bash
sbatch run_distillation_n1.sh   # 1 sample
sbatch run_distillation_n5.sh   # 5 samples
sbatch run_distillation_n10.sh  # 10 samples
sbatch run_distillation_n25.sh  # 25 samples
sbatch run_distillation_n50.sh  # 50 samples
sbatch run_distillation_n100.sh # 100 samples
```

### 4. Fine-tuning

```bash
sbatch run_finetune_ct_n1.sh    # 1 sample
sbatch run_finetune_ct_n5.sh    # 5 samples
sbatch run_finetune_ct_n10.sh   # 10 samples
sbatch run_finetune_ct_n25.sh   # 25 samples
sbatch run_finetune_ct_n50.sh   # 50 samples
sbatch run_finetune_ct_n100.sh  # 100 samples
```

### 5. Model Evaluation

To run inference on a trained model, you need to modify the `run_testing.sh` script to specify:

1. **Model Path**: Path to the trained model weights (checkpoint file)
2. **Modality**: Which modality to test on (ct, mri, or all)

```bash
# Edit the run_testing.sh script to set:
MODEL_PATH="experiments/your_experiment/checkpoints/best_model.pth"
MODALITIES="ct"  # or "mri" or "all"

# Then run:
sbatch run_testing.sh
```

### 6. Visualization

#### NIfTI Visualization:
```bash
python visualize_nifti.py \
    test_results/predictions/pred_001.nii.gz \
    datasets/resampled/test/amos_ct/labels/case_001.nii.gz \
    datasets/resampled/test/amos_ct/images/case_001.nii.gz
```

#### Results Plotting:
```bash
python scripts/plotting/plot_results.py
python scripts/plotting/plot_results_mri_baseline.py
```

### Experiment Configuration

Modify the following variables in SLURM scripts:
- `MODALITIES`: "ct", "mri", "ct,mri", or "all"
- `N_SAMPLES`: Number of samples for limited data experiments
- `BATCH_SIZE`: Batch size per GPU
- `EPOCHS`: Number of training epochs
- `LR`: Learning rate

## 📈 Results

### Expected Outputs

Each experiment generates:
- **Checkpoints**: Model weights and optimizer states
- **Training Logs**: CSV files with epoch-wise metrics
- **Training Plots**: Visualization of training curves
- **Test Results**: Segmentation predictions and metrics
- **Visualizations**: Interactive NIfTI viewers

### Metrics

The evaluation includes:
- **Dice Coefficient**: Measure of segmentation overlap
- **IoU (Jaccard)**: Intersection over Union
- **Accuracy**: Pixel-wise accuracy
- **Training Time**: Per-epoch training duration

### Reproducing Results

1. **Baseline Results**: Run `run_training.sh` with default parameters
2. **Fine-tuning Results**: Execute fine-tuning scripts with limited data
3. **Distillation Results**: Run distillation scripts with different sample sizes
4. **DANN Results**: Execute DANN scripts with varying `n_add_source` values
