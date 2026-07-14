#!/bin/bash
#SBATCH -J spectra_exp_sae_6k_cnn
#SBATCH -o spectra_exp_sae_6k_cnn.log%j
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p bsudfq
#SBATCH -t 24:00:00
echo "Date              = $(date)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated  = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated  = $SLURM_NTASKS"
echo "Number of CPUs Allocated   = $SLURM_JOB_CPUS_PER_NODE"
echo ""
set -euo pipefail

set +eu
. ~/.bashrc
set -eu
mamba activate reu26_ab

mkdir -p sae_exp

FLAG_FRAC=0.3
AT_R=0.02

# Find Python executable
which python

# Find Python version
python --version

echo "Testing Scalability for CNN..."

time python ExperimentSAE.py --seed 1 --model SixKCNN --data-root PetImages --flag-frac "$FLAG_FRAC" --at-r "$AT_R" --out-csv "sae_cnn_6k-$FLAG_FRAC-$AT_R.csv"
mv sae_*.csv sae_exp/

echo "Finished running the Scalability Experiment for the CNN."
