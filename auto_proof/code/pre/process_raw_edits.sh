#!/bin/bash
#SBATCH --job-name=raw_edits   # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --cpus-per-task=64            # 64
#SBATCH --mem=32gb                     # Job memory request (per node)
#SBATCH --time=3:00:00               # Time limit hrs:min:sec
#SBATCH -o auto_proof/data/logs/process_raw_edits/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e auto_proof/data/logs/process_raw_edits/%A_%a.err
#SBATCH --partition celltypes         # Partition used for processing

conda run -n auto_env python -m auto_proof.code.pre.process_raw_edits -n 64