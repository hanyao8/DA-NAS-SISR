#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
# source /scratch_net/zinc/wuyan/anaconda3/bin/conda shell.bash hook
conda activate pytorch
python eval.py
