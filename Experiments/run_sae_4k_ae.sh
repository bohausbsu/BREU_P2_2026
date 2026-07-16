#!/bin/bash
#SBATCH -J spectra_exp_sae_4k_ae
#SBATCH -o spectra_exp_sae_4k_ae.log%j
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

FLAG_FRAC=0.1
AT_R=0.01

# Find Python executable
which python

# Find Python version
python --version

echo "Testing Scalability for Auto-Encoder..."

for i in {1..5}; do
	python ExperimentSAE.py --seed 1 --model FourKAE --data-root mnist --flag-frac "$FLAG_FRAC" --at-r "$AT_R" --out-csv "sae_ae_4k-$i-$FLAG_FRAC-$AT_R.csv"
	mv sae_*.csv sae_exp/
done

echo "Finished running the Scalability Experiment for the Auto-Encoder."
