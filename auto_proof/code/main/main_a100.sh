#!/bin/bash
#BATCH --job-name=autoproof_a100    # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --gres=gpu:a100:1              # a100 gpu
#SBATCH -c88                  # Need cores
#SBATCH --mem=256gb                     # Job memory request (per node)
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --partition celltypes         # Partition used for processing
#SBATCH -o /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/logs/train_logs/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/logs/train_logs/%A_%a.err

conda run -n auto_env python -m auto_proof.code.main.main -n 88 -r 256