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
PRETRAINED_MODEL="/home/s2670828/multimodal_segmentation_project/experiments/exp_20250704_005127_bs1_ep100_lr0.001_wd0.0001_small/checkpoints/best_model_exp_20250704_005127_bs1_ep100_lr0.001_wd0.0001.pth"  # Replace with your model path
DATA_ROOT="/home/s2670828/multimodal_segmentation_project/datasets/resampled"  # Replace with your data path
EXPERIMENT_DIR="experiments"
BATCH_SIZE=1
EPOCHS=25
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.0001
SEED=42
MODALITIES="ct"  # Options: "ct", "mri", "ct,mri", "all"
FREEZE_ENCODER=false  # Set to true to freeze encoder layers and prevent overfitting to CT data
# FREEZE_ENCODER_EPOCH=5  # Epoch to freeze encoder (set to null or comment out to disable)

# Ablation parameters
N_SAMPLES=1   # Set to 1, 25, or 100
LOSS_FUNCTION="ce_tversky"  # Set to "combined", "ce", "dice", "tversky", or "ce_tversky"

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
echo "N_samples: $N_SAMPLES"
echo "Loss function: $LOSS_FUNCTION"

# Example ablation runs:
# 1-sample ablation (or set N_SAMPLES as needed)
accelerate launch --num_processes=2 --main_process_port 29502 main.py \
    --experiment finetune \
    --pretrained_model "$PRETRAINED_MODEL" \
    --data_root "$DATA_ROOT" \
    --experiment_dir "$EXPERIMENT_DIR/n${N_SAMPLES}_samples_finetune" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --seed $SEED \
    --modalities $MODALITIES \
    --gradient_accumulation_steps 8 \
    --mixed_precision fp16 \
    --n_samples $N_SAMPLES \
    --loss $LOSS_FUNCTION \
    $([ "$FREEZE_ENCODER" = true ] && echo "--freeze_encoder") $([ -n "$FREEZE_ENCODER_EPOCH" ] && echo "--freeze_encoder_epoch $FREEZE_ENCODER_EPOCH")

# 25-sample ablation example:
# N_SAMPLES=25
# LOSS_FUNCTION="combined"
# accelerate launch --num_processes=1 --main_process_port 29503 main.py \
#     --experiment finetune \
#     --pretrained_model "$PRETRAINED_MODEL" \
#     --data_root "$DATA_ROOT" \
#     --experiment_dir "$EXPERIMENT_DIR/n${N_SAMPLES}_samples" \
#     --batch_size $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --lr $LEARNING_RATE \
#     --weight_decay $WEIGHT_DECAY \
#     --seed $SEED \
#     --modalities $MODALITIES \
#     --gradient_accumulation_steps 8 \
#     --mixed_precision fp16 \
#     --n_samples $N_SAMPLES \
#     --loss $LOSS_FUNCTION \
#     $([ "$FREEZE_ENCODER" = true ] && echo "--freeze_encoder") $([ -n "$FREEZE_ENCODER_EPOCH" ] && echo "--freeze_encoder_epoch $FREEZE_ENCODER_EPOCH")

# 100-sample ablation example:
# N_SAMPLES=100
# LOSS_FUNCTION="ce_tversky"
# accelerate launch --num_processes=1 --main_process_port 29503 main.py \
#     --experiment finetune \
#     --pretrained_model "$PRETRAINED_MODEL" \
#     --data_root "$DATA_ROOT" \
#     --experiment_dir "$EXPERIMENT_DIR/n${N_SAMPLES}_samples" \
#     --batch_size $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --lr $LEARNING_RATE \
#     --weight_decay $WEIGHT_DECAY \
#     --seed $SEED \
#     --modalities $MODALITIES \
#     --gradient_accumulation_steps 8 \
#     --mixed_precision fp16 \
#     --n_samples $N_SAMPLES \
#     --loss $LOSS_FUNCTION \
#     $([ "$FREEZE_ENCODER" = true ] && echo "--freeze_encoder") $([ -n "$FREEZE_ENCODER_EPOCH" ] && echo "--freeze_encoder_epoch $FREEZE_ENCODER_EPOCH")

echo "Fine-tuning completed!" 