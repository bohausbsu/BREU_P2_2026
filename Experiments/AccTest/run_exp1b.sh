#!/bin/bash
#SBATCH -J spectra_exp1b_alt
#SBATCH -o spectra_exp1b_alt.log%j
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100
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

FLAG_FRAC=0.1
AT_R=0.01

for i in {1..5}; do
	echo "Run $i of 5"
	python Experiment1b.py --seed "$i" --data-root PetImages --flag-frac "$FLAG_FRAC" --at-r "$AT_R" --out-csv "cnn_alt_acc_$i.csv" --out "cnn_alt_acc_$i.png"
done

python combine_results.py --results-dir cnn_acc --out-csv "cnn_alt_acc_avg.csv" --out-png "cnn_alt_acc_avg.png" --title "SPECTRA Alt-CNN Detection Scores Averaged Over 10 Runs" --pattern "cnn_alt_acc_*.csv"
