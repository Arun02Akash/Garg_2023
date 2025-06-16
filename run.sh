#!/bin/bash
#SBATCH --job-name=run_1
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=48:00:00
#SBATCH --mem=32G    
#SBATCH --nodes=1          
#SBATCH --ntasks-per-node=8         
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load your conda setup (adjust path if needed)
source /scratch/user/<net-id>/miniconda3/etc/profile.d/conda.sh
conda activate in-context-learning
export WANDB_MODE=offline  

python src/train.py --config src/conf/toy.yaml
