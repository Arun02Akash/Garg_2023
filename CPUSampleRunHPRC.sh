#!/bin/bash
#SBATCH --job-name=aoa_10_4_5_2
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=10        
#SBATCH --partition=long

##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=arun_akashr@tamu.edu    #Send all emails to email_address

source /scratch/user/arun_akashr/miniconda3/etc/profile.d/conda.sh
conda activate pythonenv

python aoa-scipy-10-4-5-2.py