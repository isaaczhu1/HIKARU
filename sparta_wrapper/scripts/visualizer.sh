#!/bin/bash
#SBATCH -p mit_normal_gpu   
#SBATCH --gres=gpu:1 
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=10GB

module load miniforge/24.3.0-0

echo "weeee"
python sparta_wrapper/hanabi_visualizer.py