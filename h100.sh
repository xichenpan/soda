#!/bin/bash

#SBATCH --account=genai_interns
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:8
#SBATCH -t 7-00:00:00
#SBATCH --output=out
#SBATCH --error=err

export OMP_NUM_THREADS=24

echo "HOSTNAME=$HOSTNAME"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_PROCID=$SLURM_PROCID"

source ~/.bashrc
source activate base
conda activate soda

#module load cuda/12.1 \
#    nccl/2.18.3-cuda.12.1 \
#    nccl_efa/1.24.1-nccl.2.18.3-cuda.12.1

srun torchrun --standalone --nnodes=1 --nproc-per-node=8 train.py