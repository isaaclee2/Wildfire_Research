#!/bin/bash
#SBATCH --account=aoberai_286
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=output_%x-%J.out
#SBATCH --error=error_%x-%J.out
#SBATCH --job-name=fire_area_prediction
#SBATCH --mail-user=ihlee@usc.edu
#SBATCH --mail-type=ALL
module purge
#source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate diffusion_env
which python
python real_diffusion_model.py
