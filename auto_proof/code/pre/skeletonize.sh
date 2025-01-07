#!/bin/bash
#SBATCH --job-name=skeletonize    # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --cpus-per-task=1             # Only need one core
#SBATCH --mem=8gb                     # Job memory request (per node)
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
#SBATCH --array=1-24
#SBATCH -o ../../data/skeletonize_logs/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e ../../data/skeletonize_logs/%A_%a.err
#SBATCH --partition celltypes         # Partition used for processing

source conda activate gt
python skeletonize.py $SLURM_ARRAY_TASK_ID
