#!/bin/bash
#BATCH --job-name=autoproof    # Job name
#SBATCH --ntasks=1                    # Run on a single process
#SBATCH --gres=gpu:1            # blah
#SBATCH -c32               # Need cores
#SBATCH --mem=32gb                     # Job memory request (per node)
#SBATCH --time=8:00:00             # Time limit hrs:min:sec
#SBATCH -o ../../data/train_logs/%A_%a.out
# %A" is replaced by the job ID and "%a" with the array index
#SBATCH -e ../../data/train_logs/%A_%a.err
#SBATCH --partition celltypes         # Partition used for processing

source conda activate cv
python main.py