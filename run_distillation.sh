#!/bin/bash
#SBATCH --job-name=distill_unet
#SBATCH --output=logs/distill_unet_%j.out
#SBATCH --error=logs/distill_unet_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# Load modules
# module load cuda/11.8
# module load anaconda3

# Activate conda environment
source /home/${USER}/miniconda3/bin/activate diss

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${PYTHONPATH}:/home/s2670828/multimodal_segmentation_project"

# Create logs directory
mkdir -p logs

# Distillation parameters
TEACHER_MODEL="/home/s2670828/multimodal_segmentation_project/experiments/exp_20250618_012521_bs1_ep100_lr0.001_wd0.01/checkpoints/best_model_exp_20250618_012521_bs1_ep100_lr0.001_wd0.01.pth"  # Replace with your teacher model path
DATA_ROOT="/home/s2670828/multimodal_segmentation_project/datasets/resampled"  # Replace with your data path
EXPERIMENT_DIR="experiments"
BATCH_SIZE=1
EPOCHS=50
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.0001
SEED=42
MODALITIES='ct'  # Options: "ct", "mri", "ct,mri", "all"
# Add more distillation-specific options here if needed

# Run distillation with main.py orchestrator

echo "Starting knowledge distillation with main.py orchestrator..."
echo "Teacher model: $TEACHER_MODEL"
echo "Data root: $DATA_ROOT"
echo "Modalities: $MODALITIES"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Weight decay: $WEIGHT_DECAY"
echo "Seed: $SEED"

echo "Launching distillation..."

accelerate launch --num_processes=1 --main_process_port 29503 main.py \
    --experiment distill \
    --teacher_model "$TEACHER_MODEL" \
    --data_root "$DATA_ROOT" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --seed $SEED \
    --modalities $MODALITIES \
    --gradient_accumulation_steps 8 \
    --mixed_precision fp16

echo "Distillation completed!" 