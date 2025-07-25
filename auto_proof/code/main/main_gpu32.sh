#!/bin/bash
#BATCH --job-name=autoproof    # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --gres=gpu:1            # blah
#SBATCH -c32               # Need cores
#SBATCH --mem=64gb                     # Job memory request (per node)
#SBATCH --time=12:00:00             # Time limit hrs:min:sec
#SBATCH --partition celltypes         # Partition used for processing
#SBATCH -o /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/logs/train_logs/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/logs/train_logs/%A_%a.err

conda run -n auto_env python -m auto_proof.code.main.main -n 32 -r 64