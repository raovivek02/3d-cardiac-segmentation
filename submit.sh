#!/bin/bash
#SBATCH --job-name=3d_unet_heart
#SBATCH --partition=dgx
#SBATCH --output=result_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# 1. Force Conda to activate inside the SLURM job
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base



# 3. Print out the GPU hardware specs to prove it is visible
nvidia-smi

# 4. Run the script
python train.py