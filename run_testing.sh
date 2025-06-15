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

# Activate Anaconda environment
source /home/${USER}/miniconda3/bin/activate diss

# Run the testing
python test_model.py \
    --model_path "experiments/best_model.pth" \
    --test_dir "datasets/resampled/test" \
    --output_dir "test_results" 