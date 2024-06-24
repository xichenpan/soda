#!/bin/bash
#SBATCH --account=genai_interns
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH -t 7-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

sleep infinity