#!/bin/bash
#SBATCH --job-name=process_segclr    # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --cpus-per-task=1            # 16
#SBATCH --mem=16gb                     # Job memory request (per node)
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --array=1-24
#SBATCH -o /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/logs/process_segclr/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/logs/process_segclr/%A_%a.err
#SBATCH --partition celltypes         # Partition used for processing

conda run -n train python -m auto_proof.code.pre.process_segclr -c $SLURM_ARRAY_TASK_ID -n 16