#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1    # nodes requested
#SBATCH -n 1    # tasks requested
#SBATCH --gres=gpu:1  # use 2 GPUs
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 12:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# SCRATCH_DISK=/disk/scratch
# dest_path=${SCRATCH_DISK}/${USER}/multimodal_segmentation_project/

# Default values
DATA_ROOT=/home/s2670828/multimodal_segmentation_project/datasets/resampled #${dest_path}/datasets/resampled
BATCH_SIZE=1  # Keep batch size per GPU the same
EPOCHS=100
LR=0.001  # Reduced learning rate further
EXPERIMENT_DIR="experiments"
GRAD_ACCUM_STEPS=8  # Doubled gradient accumulation to compensate for fewer GPUs
WEIGHT_DECAY=0.0001  # Default weight decay; change as needed
DROPOUT_RATE=0.1  # Default dropout rate; change as needed
LAMBDA_DOMAIN=0.2  # Reduced further to prevent NaN errors
SOURCE_MODALITY="all"
TARGET_MODALITY="ct"
N_SAMPLES=100
N_ADD_SOURCE=25  # Number of additional source volumes from add/
N_TARGET=125    # Number of target volumes from target/

# Pre-trained model path
PRETRAINED_MODEL="experiments/exp_20250704_005127_bs1_ep100_lr0.001_wd0.0001_small/checkpoints/best_model_exp_20250704_005127_bs1_ep100_lr0.001_wd0.0001.pth"

# Create directories if they don't exist
mkdir -p $EXPERIMENT_DIR

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate diss

# ====================
# Clean up scratch space
# ====================
# if [ -d "${dest_path}" ]; then
#     echo "Deleting scratch disk path: ${dest_path}"
#     rm -rf "${dest_path}"
# else
#     echo "Scratch disk path does not exist: ${dest_path}"
# fi

# mkdir -p ${dest_path}
# src_path="/home/${USER}/multimodal_segmentation_project/"
# rsync -azvP \
#   --exclude='experiments/' \
#   --exclude='logs/' \
#   --exclude='test_results/' \
#   "${src_path}" "${dest_path}"

# Run the DANN training
python main.py \
    --experiment dann \
    --source_modality $SOURCE_MODALITY \
    --target_modality $TARGET_MODALITY \
    --data_root $DATA_ROOT \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --experiment_dir $EXPERIMENT_DIR \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --mixed_precision fp16 \
    --dropout_rate $DROPOUT_RATE \
    --lambda_domain $LAMBDA_DOMAIN \
    --pretrained_model $PRETRAINED_MODEL \
    --n_add_source $N_ADD_SOURCE \
    --n_target $N_TARGET \
    --early_stopping \
    --patience 25 
    # --n_samples $N_SAMPLES
    

# Print completion message
echo "DANN training completed! Check $EXPERIMENT_DIR for experiment results." 