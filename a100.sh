#!/bin/bash

#SBATCH --account=genai_interns
#SBATCH --qos=genai_interns
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH -t 14-00:00:00
#SBATCH --output=out.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export NCCL_BUFFSIZE=2097152
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

echo "HOSTNAME=$HOSTNAME"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_PROCID=$SLURM_PROCID"

source ~/.bashrc
source activate base
conda activate soda

#srun accelerate launch \
#    --multi_gpu \
#    --num_processes=16 \
#    --num_machines=2 \
#    --machine_rank=$SLURM_PROCID \
#    --main_process_ip=$HOSTNAME \
#    --main_process_port=29500 \
#    --mixed_precision=bf16 \
#    train.py

srun torchrun --nnodes=4 --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29500 train.py