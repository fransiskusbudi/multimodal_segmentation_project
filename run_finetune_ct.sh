#!/bin/bash
#SBATCH --job-name=finetune_ct
#SBATCH --output=logs/finetune_ct_%j.out
#SBATCH --error=logs/finetune_ct_%j.err
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

# Fine-tuning parameters
PRETRAINED_MODEL="/home/s2670828/multimodal_segmentation_project/experiments/exp_20250618_012521_bs1_ep100_lr0.001_wd0.01/checkpoints/best_model_exp_20250618_012521_bs1_ep100_lr0.001_wd0.01.pth"  # Replace with your model path
DATA_ROOT="/home/s2670828/multimodal_segmentation_project/datasets/resampled"  # Replace with your data path
EXPERIMENT_DIR="experiments"
BATCH_SIZE=1
EPOCHS=15
LEARNING_RATE=0.00001
WEIGHT_DECAY=0.05
SEED=42
MODALITIES="ct"  # Options: "ct", "mri", "ct,mri", "all"
FREEZE_ENCODER=true  # Set to true to freeze encoder layers and prevent overfitting to CT data
FREEZE_ENCODER_EPOCH=5  # Epoch to freeze encoder (set to null or comment out to disable)

# Run fine-tuning with main.py orchestrator
echo "Starting fine-tuning with main.py orchestrator..."
echo "Pre-trained model: $PRETRAINED_MODEL"
echo "Data root: $DATA_ROOT"
echo "Modalities: $MODALITIES"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Freeze encoder: $FREEZE_ENCODER"
echo "Freeze encoder epoch: $FREEZE_ENCODER_EPOCH"

python main.py \
    --experiment finetune \
    --pretrained_model "$PRETRAINED_MODEL" \
    --data_root "$DATA_ROOT" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --seed $SEED \
    --modalities $MODALITIES \
    --gradient_accumulation_steps 2 \
    --mixed_precision fp16 \
    $([ "$FREEZE_ENCODER" = true ] && echo "--freeze_encoder") \
    $([ -n "$FREEZE_ENCODER_EPOCH" ] && echo "--freeze_encoder_epoch $FREEZE_ENCODER_EPOCH")

echo "Fine-tuning completed!" 