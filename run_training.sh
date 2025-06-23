#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 12:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# Default values
DATA_ROOT="datasets/resampled"
BATCH_SIZE=2
EPOCHS=50
LR=0.001
EXPERIMENT_DIR="experiments"
MODALITIES="all"  # Options: "all", "ct", "mri", "ct,mri"

# Create experiment directory if it doesn't exist
mkdir -p $EXPERIMENT_DIR

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate diss

SCRATCH_DISK=/disk/scratch
dest_path=${SCRATCH_DISK}/${USER}/multimodal_segmentation_project/

# ====================
# Clean up scratch space
# ====================
if [ -d "${dest_path}" ]; then
    echo "Deleting scratch disk path: ${dest_path}"
    rm -rf "${dest_path}"
else
    echo "Scratch disk path does not exist: ${dest_path}"
fi

mkdir -p ${dest_path}
src_path=/home/${USER}/multimodal_segmentation_project/
rsync -azvP ${src_path} ${dest_path}

# Run the training
python train_unet.py \
    --data_root $DATA_ROOT \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --experiment_dir $EXPERIMENT_DIR \
    --modalities $MODALITIES

echo "Deleting scratch disk path: ${dest_path}"
rm -rf ${dest_path}

# Print completion message
echo "Training completed! Check $EXPERIMENT_DIR for experiment results." 
 