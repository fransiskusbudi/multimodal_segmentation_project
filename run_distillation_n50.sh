#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:2  # use 2 GPUs
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard-Noble
#SBATCH -t 12:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.
#SBATCH --nodelist=damnii06

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
TEACHER_MODEL="/home/s2670828/multimodal_segmentation_project/experiments/exp_20250801_002447_bs1_ep100_lr0.001_wd0.0001_baseline_mri/checkpoints/best_model_exp_20250801_002447_bs1_ep100_lr0.001_wd0.0001.pth"  # Replace with your teacher model path
DATA_ROOT="/home/s2670828/multimodal_segmentation_project/datasets/resampled"  # Replace with your data path
EXPERIMENT_DIR="experiments"
BATCH_SIZE=1
EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
SEED=42
MODALITIES='ct'  # Options: "ct", "mri", "ct,mri", "all"
# Early stopping parameters
EARLY_STOPPING=false  # Set to true to enable early stopping
PATIENCE=10  # Number of epochs to wait for improvement before stopping
# Distillation loss parameters
ALPHA=0.7  # Weight for segmentation loss in distillation
TEMPERATURE=2.0  # Temperature for softening logits in distillation
# Ablation study parameters
N_SAMPLES=50  # Number of samples to use for training (set to None to use all samples)
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
echo "Early stopping: $EARLY_STOPPING"
echo "Patience: $PATIENCE"
echo "Alpha: $ALPHA"
echo "Temperature: $TEMPERATURE"
echo "Number of samples: $N_SAMPLES"

echo "Launching distillation..."

accelerate launch --num_processes=2 --main_process_port 29502 main.py \
    --experiment distill \
    --teacher_model "$TEACHER_MODEL" \
    --data_root "$DATA_ROOT" \
    --experiment_dir "$EXPERIMENT_DIR/n${N_SAMPLES}_samples_distill" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --seed $SEED \
    --modalities $MODALITIES \
    --gradient_accumulation_steps 8 \
    --mixed_precision fp16 \
    --alpha $ALPHA \
    --temperature $TEMPERATURE \
    --n_samples $N_SAMPLES \
    # --early_stopping \
    # --patience $PATIENCE \

echo "Distillation completed!" 