#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# export CUDA_VISIBLE_DEVICES=1
# rm -r results/trial_run_pathtracker_ffstlstm/
# python mainclean.py --print-freq 20 --lr 1e-03 --epochs 1000 -b 16 --algo rbp --model ffstlstm --name trial_run_pathtracker_ffstlstm --log #--parallel

# python mainclean.py --print-freq 20 --lr 1e-03 --epochs 10 -b 2 --algo rbp --model ff --name trial_run_pathtracker_ff_10_epoch --log #--parallel

# export CUDA_VISIBLE_DEVICES=2
# rm -r results/trial_run_pathtracker_ffhgru_gabor/
# python mainclean.py --print-freq 20 --lr 1e-03 --epochs 10 -b 2 --algo rbp --model ffhgru --name trial_run_pathtracker_ffhgru_gabor --log #--parallel

# export CUDA_VISIBLE_DEVICES=1,2
rm -r results/trial_run_pathtracker_ffhgru2d_gaussian_5_2_parallel_no_dropout_hgru_dim_4/
CUDA_VISIBLE_DEVICES=1,2 python mainclean.py --print-freq 20 --lr 1e-04 --epochs 300 -b 32 --algo rbp --model ffhgru --name trial_run_pathtracker_ffhgru2d_gaussian_5_2_parallel_no_dropout_hgru_dim_4 --log --parallel


# rm -r results/trial_run_pathtracker_ffhgru2d_gaussian_5_2_parallel_no_dropout_hgru_dim_4/
rm -r results/test_drew
CUDA_VISIBLE_DEVICES=5,6 python mainclean.py --print-freq 20 --lr 1e-04 --epochs 300 -b 160 --algo rbp --model clock_dynamic --name test_drew --log --parallel

