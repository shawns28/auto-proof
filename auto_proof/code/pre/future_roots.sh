#!/bin/bash
#BATCH --job-name=skeletonize    # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --cpus-per-task=32            # Need cores
#SBATCH --mem=8gb                     # Job memory request (per node)
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
#SBATCH --array=1-24
#SBATCH -o /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/logs/future_roots_logs/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/logs/future_roots_logs/%A_%a.err
#SBATCH --partition celltypes         # Partition used for processing

conda run -n train python -m auto_proof.code.pre.future_roots -c $SLURM_ARRAY_TASK_ID
