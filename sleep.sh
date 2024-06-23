#!/bin/bash
#SBATCH --account=genai_interns
#SBATCH --qos=genai_interns
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -t 14-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

sleep infinity