#!/bin/bash
#SBATCH -J spectra_exp1b
#SBATCH -o spectra_exp1b.log%j
#SBATCH --cpus-per-task=7
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -t 24:00:00
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated  = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated  = $SLURM_NTASKS"
echo "Number of CPUs Allocated   = $SLURM_JOB_CPUS_PER_NODE"
echo "GPU Allocated              = $SLURM_JOB_GPUS"
echo ""
set -euo pipefail

set +eu
. ~/.bashrc
set -eu
mamba activate reu26_ab

mkdir -p cnn_acc

FLAG_FRAC=0.3
AT_R=0.02

for i in {1..10}; do
	echo "Run $i of 10"
	python Experiment1b.py --seed "$i" --data-root PetImages --flag-frac "$FLAG_FRAC" --at-r "$AT_R" --out-csv "cnn_acc_$i-$FLAG_FRAC-$AT_R.csv" --out "cnn_acc_$i-$FLAG_FRAC-$AT_R.png"
	mv cnn_acc_* cnn_acc/
done

python combine_results.py --results-dir cnn_acc --out-csv "cnn_acc/cnn_acc_avg_$FLAG_FRAC-$AT_R.csv" --out-png "cnn_acc/cnn_acc_avg_$FLAG_FRAC-$AT_R.png" --title "SPECTRA CNN Detection Scores Averaged Over 10 Runs" --pattern "cnn_acc_*.csv"
