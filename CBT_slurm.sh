#!/bin/bash
#SBATCH --job-name=ms_CB             # Assign an short name to your job
#SBATCH --cpus-per-task=2            # Cores per task (>1 if multithread tasks)
#SBATCH -G 1
#SBATCH --array=0-1
#SBATCH --mem=50000                  # Real memory (RAM) required (MB)
#SBATCH --time=48:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%A_%a-%j.out     # STDOUT output file
#SBATCH --error=slurm.%N.%A_%a-%j.err      # STDERR output file (optional)
echo "Resource allocated"
nvidia-smi
# activate my own environment
source /common/home/joshcoop/miniconda3/bin/activate torch
echo "Directory changed"
echo "Starting CBT Run"
python src/experiment_run.py $SLURM_ARRAY_TASK_ID
echo "CBT Run Complete"
