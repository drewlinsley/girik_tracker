CUDA_VISIBLE_DEVICES=0 python generate_mturk_preds.py --print-freq 20 --lr 1e-03 --epochs 300 --model ffhgru --name hgru_wider_32 --parallel --length=64 --speed=1 --dist=14
CUDA_VISIBLE_DEVICES=0 python generate_mturk_preds.py --print-freq 20 --lr 1e-03 --epochs 300 --model ffhgru_v2 --name hgru_v2 --parallel --length=64 --speed=1 --dist=14
CUDA_VISIBLE_DEVICES=0 python generate_mturk_preds.py --print-freq 20 --lr 1e-03 --epochs 300 --model ffhgru_no_inh --name ffhgru_no_inh --parallel --length=64 --speed=1 --dist=14
CUDA_VISIBLE_DEVICES=0 python generate_mturk_preds.py --print-freq 20 --lr 1e-03 --epochs 300 --model ffhgru_no_mult --name ffhgru_no_mult --parallel --length=64 --speed=1 --dist=14
CUDA_VISIBLE_DEVICES=0 python generate_mturk_preds.py --print-freq 20 --lr 1e-03 --epochs 300 --model ffhgru_no_add --name ffhgru_no_add --parallel --length=64 --speed=1 --dist=14
CUDA_VISIBLE_DEVICES=0 python generate_mturk_preds.py --print-freq 20 --lr 1e-03 --epochs 300 --model ffhgru_mult_add --name ffhgru_mult_add --parallel --length=64 --speed=1 --dist=14

