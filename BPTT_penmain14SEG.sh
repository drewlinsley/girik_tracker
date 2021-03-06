#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
python mainclean.py data/train_images_14_SEG.txt data/test_images_14_SEG.txt --print-freq 20 --lr 1e-03 --epochs 20 -b 128 --algo bptt --penalty --name BPTT_penalty_hgru_PF14SEG_6ts_4GPU_128Batch --parallel --log
