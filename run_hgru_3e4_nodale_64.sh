#!/bin/bash

#SBATCH --time=60:00:00
#SBATCH -p gpu --gres=gpu:4
#SBATCH --mem=24G
#SBATCH -n 4
#SBATCH -N 4
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
#SBATCH -J pathtracer

# Specify an output file
# #SBATCH -o perf.out
# #SBATCH -e perf.err

source activate newtrack
module load gcc/8.3
module load cuda/10.2

export PYTHONPATH=/users/akarkada/axial-positional-embedding:$PYTHONPATH
export PYTHONPATH=/users/akarkada/Pytorch-Correlation-extension:$PYTHONPATH
python mainclean.py --print-freq 20 --lr 3e-04 --epochs 2000 -b 180 --model ffhgru_tanh --name ffhgru_tanh_3e-4 --log --length 64 --speed 1 --dist 14 --parallel

