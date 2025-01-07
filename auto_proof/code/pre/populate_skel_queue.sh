#!/bin/bash
#SBATCH --job-name=populate_skel    # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --cpus-per-task=1             # Only need one core
#SBATCH --mem=4gb                     # Job memory request (per node)
#SBATCH --time=16:00:00               # Time limit hrs:min:sec
#SBATCH --array=1-8
#SBATCH -o ../../data/populate_skel_queue_logs/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e ../../data/populate_skel_queue_logs/%A_%a.err
#SBATCH --partition celltypes         # Partition used for processing

source conda activate auto-proof
python populate_skel_queue.py $SLURM_ARRAY_TASK_ID
