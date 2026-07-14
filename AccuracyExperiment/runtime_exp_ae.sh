#!/bin/bash
#SBATCH -J spectra_runtime_exp_ae
#SBATCH -o spectra_runtime_exp_ae.log%j
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

which python

nvidia-smi

python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.get_device_properties())"

python ExperimentRuntime.py --model ae --dataset-root mnist --out-csv "runtime/ae_runtime.csv"

