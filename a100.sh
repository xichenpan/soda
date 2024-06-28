#!/bin/bash

#SBATCH --account=genai_interns
#SBATCH --qos=genai_interns
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH -t 14-00:00:00
#SBATCH --output=out
#SBATCH --error=err

export OMP_NUM_THREADS=12
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="N-M"

export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_SOCKET_IFNAME=ens32

echo "HOSTNAME=$HOSTNAME"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_PROCID=$SLURM_PROCID"

source ~/.bashrc
source activate base
conda activate soda

module load cuda/12.1 \
    nccl/2.18.3-cuda.12.1 \
    nccl_efa/1.24.1-nccl.2.18.3-cuda.12.1

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
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29500 train.py \
    --output_dir /fsx-project/xichenpan/output \
    --data_dir /fsx-project/xichenpan/.cache \
    --logging_dir /fsx-project/xichenpan/logging