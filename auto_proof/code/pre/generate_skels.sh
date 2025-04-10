#!/bin/bash
#SBATCH --job-name=generate_skel    # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --cpus-per-task=1             # Only one core otherwise overloads skeleton service
#SBATCH --mem=8gb                     # Job memory request (per node)
#SBATCH --time=36:00:00               # Time limit hrs:min:sec
#SBATCH -o /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/logs/generate_skels/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e /allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/test_data/logs/generate_skels/%A_%a.err
#SBATCH --partition celltypes         # Partition used for processing

conda run -n train python -m auto_proof.code.pre.generate_skels