#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:2  # use 2 GPUs
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
DATA_ROOT=/home/s2670828/multimodal_segmentation_project/datasets/resampled
BATCH_SIZE=1  # Keep batch size per GPU the same
EPOCHS=100
LR=0.001
EXPERIMENT_DIR="experiments"
GRAD_ACCUM_STEPS=8  # Doubled gradient accumulation to compensate for fewer GPUs
MODALITIES="ct"  # Default to all modalities; change as needed
WEIGHT_DECAY=0.0001  # Default weight decay; change as needed
DROPOUT_RATE=0.1  # Default dropout rate; change as needed
N_SAMPLES=1  # Number of samples to use for training (first 100 samples)

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

# accelerate launch --num_processes=2 --main_process_port 29503 src/main.py --train

# Run the training
accelerate launch --num_processes=2 --main_process_port 29502 main.py \
    --experiment train \
    --data_root $DATA_ROOT \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --experiment_dir $EXPERIMENT_DIR \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --mixed_precision fp16 \
    --loss ce_tversky \
    --modalities $MODALITIES \
    --dropout_rate $DROPOUT_RATE \
    --n_samples $N_SAMPLES \
    # --early_stopping \
    # --patience 25 \

# echo "Deleting scratch disk path: ${dest_path}"
# rm -rf ${dest_path}

# Print completion message
echo "Training completed! Check $EXPERIMENT_DIR for experiment results." 
