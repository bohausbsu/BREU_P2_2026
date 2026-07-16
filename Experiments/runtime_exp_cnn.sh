#!/bin/bash
#SBATCH -J spectra_runtime_exp_cnn
#SBATCH -o spectra_runtime_exp_cnn.log%j
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

mkdir -p runtime

FLAG_FRAC=0.1
AT_R=0.01

which python

nvidia-smi

python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.get_device_properties())"

for i in {1..5}; do
	python ExperimentRuntime.py --model cnn --dataset-root PetImages --flag-frac "$FLAG_FRAC" --at-r "$AT_R" --out-csv "runtime/cnn_$i-runtime.csv"
done
