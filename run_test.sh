CUDA_VISIBLE_DEVICES=4 python mainclean.py --print-freq 20 --lr 1e-03 --epochs 300 -b 128 --model ffhgru --name test_drew --log --parallel --ckpt results/test_drew/saved_models/model_fscore_4957_epoch_125_checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=3 python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 -b 128 --model ffhgru --name test_drew --log --parallel --ckpt=results/hgru_wider_32/saved_models/model_fscore_5521_epoch_122_checkpoint.pth.tar



CUDA_VISIBLE_DEVICES=4,5,6,7 python mainclean.py --print-freq 20 --lr 1e-03 --epochs 300 -b 180 --model r3d --name imagenet_r3d_1e-3 --log --parallel --pretrained
CUDA_VISIBLE_DEVICES=1,2,3,0 python mainclean.py --print-freq 20 --lr 1e-02 --epochs 300 -b 180 --model r3d --name imagenet_r3d_1e-2 --log --parallel --pretrained






CUDA_VISIBLE_DEVICES=4,5,6,7 python mainclean.py --print-freq 20 --lr 3e-04 --epochs 300 -b 196 --model ffhgru --name hgru --log --parallel


CUDA_VISIBLE_DEVICES=4,5,6,7 python mainclean.py --print-freq 20 --lr 1e-3 --epochs 1000 -b 280 --model ffhgru --name hgru_v1 --log --parallel


CUDA_VISIBLE_DEVICES=0,1,2,3 python mainclean.py --print-freq 20 --lr 1e-03 --epochs 300 -b 180 --model r2plus1 --name  r2plus_1e-4 --log --parallel
CUDA_VISIBLE_DEVICES=4,5,6,7 python mainclean.py --print-freq 20 --lr 1e-03 --epochs 300 -b 180 --model r2plus1 --name  r2plus_1e-3 --log --parallel
CUDA_VISIBLE_DEVICES=4,5,6,7 python mainclean.py --print-freq 20 --lr 1e-02 --epochs 300 -b 180 --model r2plus1 --name  r2plus_1e-2 --log --parallel


CUDA_VISIBLE_DEVICES=0,1,2,3 python mainclean.py --print-freq 20 --lr 1e-03 --epochs 300 -b 180 --model nostride_r3d --name  nostride_r3d_1e-3 --log --parallel



CUDA_VISIBLE_DEVICES=4,5,6,7 python mainclean.py --print-freq 20 --lr 1e-3 --epochs 1000 -b 280 --model ffhgru_v2 --name hgru_v2 --log --parallel




CUDA_VISIBLE_DEVICES=7 python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 -b 32 --model ffhgru --name test_drew --log --parallel --ckpt results/hgru_wider_32/saved_models/model_val_acc_0092_epoch_1271_checkpoint.pth.tar
