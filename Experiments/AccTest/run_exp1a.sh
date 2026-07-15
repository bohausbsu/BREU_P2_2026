#!/bin/bash
#SBATCH -J spectra_exp1a_alt
#SBATCH -o spectra_exp1a_alt.log%j
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100
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

FLAG_FRAC=0.1
AT_R=0.01
EFF_SIGNAL_RATIO=0.3

which python
for i in {1..5}; do
	echo "Run $i of 5"
	python Experiment1a.py \
		--seed "$i" \
		--dataset college_student_placement_dataset.csv \
		--target-col Placement \
		--flag-frac "$FLAG_FRAC" \
		--at-r "$AT_R" \
		--eff-signal-ratio "$EFF_SIGNAL_RATIO" \
		--out-csv "ffnn_alt_acc_$i-$EFF_SIGNAL_RATIO.csv" \
		--out "ffnn_alt_acc_$i-$EFF_SIGNAL_RATIO.png"
done

python combine_results.py \
	--results-dir ffnn_acc \
	--out-csv "ffnn_alt_acc_avg-$EFF_SIGNAL_RATIO.csv" \
	--out-png "ffnn_alt_acc_avg-$EFF_SIGNAL_RATIO.png" \
	--title "SPECTRA Alt FFNN Detection Scores at $EFF_SIGNAL_RATIO" \
	--pattern "ffnn_alt_acc_*.csv"
