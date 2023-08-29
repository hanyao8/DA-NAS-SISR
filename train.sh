#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:2
#SBATCH  --mem=50G
#SBATCH  --constraint='titan_xp'
# source /scratch_net/zinc/wuyan/anaconda3/bin/conda shell.bash hook
conda activate pytorch
python train.py
