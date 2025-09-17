#!/bin/bash
#SBATCH --partition=debug
eval "$(conda shell.bash hook)"
conda activate diffusion_env
which python
python -c "import torch"
