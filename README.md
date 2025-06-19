# Multimodal Medical Image Segmentation

This project implements a deep learning-based approach for medical image segmentation using multimodal data. The implementation uses PyTorch and is designed to work with 3D medical images.

## Project Structure

```
multimodal_segmentation_project/
├── models/
│   └── unet.py           # U-Net model implementation
├── utils/
│   ├── dataloader.py     # Data loading and preprocessing
│   └── metrics.py        # Evaluation metrics
├── notebooks/
│   ├── testing.ipynb     # Testing and 
visualization
│   ├── viz_amos.ipynb    # AMOS dataset visualization
│   └── spacing.ipynb     # Spacing analysis
├── train_unet.py         # Training script
├── test_model.py         # Testing script
├── main.py              # Main orchestration script
└── run_training.sh      # SLURM script for training
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/fransiskusbudi/multimodal_segmentation_project.git
cd multimodal_segmentation_project
```

2. Create and activate a conda environment:
```bash
conda create -n diss python=3.8
conda activate diss
```

3. Install dependencies:
```bash
pip install torch torchvision
pip install nibabel matplotlib numpy
pip install accelerate
```

## Usage

### Training

To train the model, use the provided SLURM script:
```bash
sbatch run_training.sh
```

The script uses the following default parameters:
- Batch size: 1
- Learning rate: 0.001
- Epochs: 50
- Mixed precision: fp16
- Gradient accumulation steps: 8

### Testing

To test the model:
```bash
sbatch run_testing.sh
```

This will:
1. Load the trained model
2. Run inference on test data
3. Save predictions as NIfTI files
4. Generate visualizations

## Model Architecture

The project uses a 3D U-Net architecture with the following features:
- 3D convolutions for volumetric data processing
- Skip connections for better feature preservation
- Batch normalization for stable training
- Leaky ReLU activation functions

## Data

The project is designed to work with 3D medical images in NIfTI format. The data should be organized as follows:

```
datasets/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 