#!/bin/bash
#BATCH --job-name=labels    # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --cpus-per-task=1            # Need cores
#SBATCH --mem=8gb                     # Job memory request (per node)
#SBATCH --time=16:00:00               # Time limit hrs:min:sec
#SBATCH --array=1-24
#SBATCH -o ../../data/labels_logs/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e ../../data/labels_logs/%A_%a.err
#SBATCH --partition celltypes         # Partition used for processing

source conda activate gt
python labels.py $SLURM_ARRAY_TASK_ID