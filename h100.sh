#!/bin/bash

#SBATCH --account=genai_interns
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH -t 7-00:00:00
#SBATCH --output=out.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LOCAL_RANK=$SLURM_PROCID

echo "HOSTNAME=$HOSTNAME"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_PROCID=$SLURM_PROCID"

conda init
source ~/.bashrc
source activate base
conda activate soda

srun torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29500 train.py --output_dir ~/output --data_dir ~/.cache