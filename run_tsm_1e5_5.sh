#!/bin/bash

#SBATCH --time=60:00:00
#SBATCH -p gpu --gres=gpu:6
#SBATCH -n 6
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
#SBATCH -J pathtracer

# Specify an output file
# #SBATCH -o time.out
# #SBATCH -e time.err

source activate newtrack
module load gcc/8.3
module load cuda/10.2

export PYTHONPATH=/users/akarkada/axial-positional-embedding:$PYTHONPATH
export PYTHONPATH=/users/akarkada/Pytorch-Correlation-extension:$PYTHONPATH
python mainclean.py --print-freq 20 --lr 1e-05 --epochs 2000 -b 56 --model rntsm --name rntsm_1e-5 --log --parallel --length 64 --speed 1 --dist 5

