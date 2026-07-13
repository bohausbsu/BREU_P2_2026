#!/bin/bash
#SBATCH -J spectra_exp1a
#SBATCH -o spectra_exp1a.log%j
#SBATCH --cpus-per-task=7
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -t 24:00:00
echo "Date              = $(date)"
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

mkdir -p ffnn_acc

FLAG_FRAC=0.3
AT_R=0.02

which python
for i in {1..1}; do
	echo "Run $i of 10"
	python Experiment1a.py --seed "$i" --dataset college_student_placement_dataset.csv --target-col Placement --flag-frac "$FLAG_FRAC" --at-r "$AT_R" --out-csv "ffnn_acc_$i-$FLAG_FRAC-$AT_R.csv" --out "ffnn_acc_$i-$FLAG_FRAC-$AT_R.png"
	mv ffnn_acc_* ffnn_acc/
done

python combine_results.py --results-dir ffnn_acc --out-csv "ffnn_acc/ffnn_acc_avg_$FLAG_FRAC-$AT_R.csv" --out-png "ffnn_acc/ffnn_acc_avg_$FLAG_FRAC-$AT_R.png" --title "SPECTRA FFNN Detection Scores Averaged Over 10 Runs" --pattern "ffnn_acc_*.csv"
