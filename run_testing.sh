#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU for testing
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 2:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# Default values
# MODEL_PATH="experiments/finetune_20250626_174347_best_model_exp_20250618_012521_bs1_ep100_lr0/checkpoints/best_finetuned_model_finetune_20250626_174347_best_model_exp_20250618_012521_bs1_ep100_lr0.pth"
MODEL_PATH="experiments/exp_20250704_005127_bs1_ep100_lr0.001_wd0.0001/checkpoints/best_model_exp_20250704_005127_bs1_ep100_lr0.001_wd0.0001.pth"
DATA_ROOT="datasets/resampled"
EXPERIMENT_DIR="test_results"
MODEL_NAME="unet"
MODALITIES="ct"  # Options: "all", "ct", "mri", "ct,mri"

# Activate Anaconda environment
source /home/${USER}/miniconda3/bin/activate diss

# Run the evaluation with main.py orchestrator
echo "Starting evaluation with main.py orchestrator..."
echo "Model path: $MODEL_PATH"
echo "Data root: $DATA_ROOT"
echo "Modalities: $MODALITIES"
echo "Model name: $MODEL_NAME"

python main.py \
    --experiment eval \
    --model_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --experiment_dir $EXPERIMENT_DIR \
    --model_name $MODEL_NAME \
    --modalities $MODALITIES

echo "Evaluation completed!"