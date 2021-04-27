CUDA_VISIBLE_DEVICES=0 python measure_model_tuning.py --print-freq 20 --lr 1e-03 --epochs 300 --model ffhgru --name hgru_wider_32 --parallel --length=64 --speed=1 --dist=14
