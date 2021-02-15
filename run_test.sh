CUDA_VISIBLE_DEVICES=4 python mainclean.py --print-freq 20 --lr 1e-03 --epochs 300 -b 128 --model ffhgru --name test_drew --log --parallel --ckpt results/test_drew/saved_models/model_fscore_4957_epoch_125_checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=3 python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 -b 128 --model ffhgru --name test_drew --log --parallel --ckpt=results/hgru_wider_32/saved_models/model_fscore_5521_epoch_122_checkpoint.pth.tar



python mainclean.py --print-freq 20 --lr 1e-03 --epochs 300 -b 128 --model 3d --name r3d --log --parallel


CUDA_VISIBLE_DEVICES=4,5,6,7 python mainclean.py --print-freq 20 --lr 3e-04 --epochs 300 -b 196 --model ffhgru --name hgru --log --parallel


CUDA_VISIBLE_DEVICES=4,5,6,7 python mainclean.py --print-freq 20 --lr 1e-3 --epochs 1000 -b 280 --model ffhgru --name hgru_v1 --log --parallel




