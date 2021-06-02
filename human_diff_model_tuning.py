#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:04:57 2019

"""

import os
import time
import torch
from torchvision.transforms import Compose as transcompose
import torch.nn.parallel
from torch import nn
import torch.optim
import numpy as np

# from utils.dataset import DataSetSeg
from utils import engine
from utils.TFRDataset import tfr_data_loader
from models.hgrucleanSEG import hConvGRU
from models.FFnet import FFConvNet
from models.ffhgru import FFhGRU  # , FFhGRUwithGabor, FFhGRUwithoutGaussian, FFhGRUdown
from models import ffhgru

from utils.transforms import GroupScale, Augmentation, Stack, ToTorchFormatTensor
from utils.misc_functions import AverageMeter, FocalLoss, acc_scores, save_checkpoint
from statistics import mean
from utils.opts import parser
from utils import presets
import matplotlib
# import imageio
from torch._six import inf
from torchvideotransforms import video_transforms, volume_transforms
from torchvision.models import video
from models import nostridetv as nostride_video
from tqdm import tqdm
from types import SimpleNamespace
from glob import glob
from matplotlib import pyplot as plt


torch.backends.cudnn.benchmark = True
args = parser.parse_args()
video_transform_list = [video_transforms.RandomHorizontalFlip(0.5), video_transforms.RandomVerticalFlip(0.5)]  # , volume_transforms.ClipToTensor(div_255=False)]
transforms = video_transforms.Compose(video_transform_list)
use_augmentations = False
disentangle_channels = False
plot_incremental = False
debug_data = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_best_model(directory, model, prep_gifs=3, batch_size=40):
    """Given a directory, find the best performing checkpoint and evaluate it on all datasets."""
    # perfs = np.load(os.path.join(directory, "val.npz"))["loss"]
    # arg_perf = np.argmin(perfs)
    perfs = np.load(os.path.join(directory, "val.npz"))["balacc"]
    arg_perf = np.argmax(perfs)
    weights = glob(os.path.join(directory, "saved_models", "*.tar"))
    weights.sort(key=os.path.getmtime)
    weights = np.asarray(weights)
    ckpt = weights[arg_perf]

    # Construct new args
    args = SimpleNamespace()
    args.batch_size = batch_size
    args.parallel = True
    args.ckpt = ckpt
    args.model = model
    args.penalty = "Testing"
    args.algo = "Testing"
    if "imagenet" in directory:
        args.pretrained = True
    else:
        args.pretrained = False
    ds = engine.get_datasets()
    for d in ds:
        evaluate_model(results_folder, args, prep_gifs=prep_gifs, dist=d["dist"], speed=d["speed"], length=d["length"])


def evaluate_model(results_folder, args, prep_gifs=3, dist=14, speed=1, length=64, height=32, width=32):
    """Evaluate a model and plot results."""
    os.makedirs(results_folder, exist_ok=True)
    model = engine.model_selector(args=args, timesteps=length, device=device)

    pf_root, timesteps, len_train_loader, len_val_loader = engine.human_dataset_selector(set_name="gen_1_25_64")
    print("Loading training dataset")
    train_loader = tfr_data_loader(data_dir=os.path.join(pf_root,'train-*'), batch_size=args.batch_size, drop_remainder=True, timesteps=timesteps, height=height, width=width, shuffle_buffer=0)

    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    if args.parallel is True:
        model = torch.nn.DataParallel(model).to(device)
        print("Loading parallel finished on GPU count:", torch.cuda.device_count())
    else:
        model = model.to(device)
        print("Loading finished")

    # noqa Save timesteps/kernel_size/dimensions/learning rate/epochs/exp_name/algo/penalty to a dict for reloading in the future
    param_names_shapes = {k: v.shape for k, v in model.named_parameters()}
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    print("Including parameters {}".format([k for k, v in model.named_parameters()]))

    assert args.ckpt is not None, "You must pass a checkpoint for testing."
    model = engine.load_ckpt(model, args.ckpt)

    model.eval()
    accs = []
    losses = []
    for epoch in range(1):

        time_since_last = time.time()
        # model.train()
        end = time.perf_counter()

        loader = train_loader
        human_data = np.load("mturk_responses/exp4_64_26_average_responses.npy")[:, 1].ravel().astype(np.float32)
        human_data = torch.from_numpy(human_data).to("cuda")
        for idx, (imgs, target) in tqdm(enumerate(loader), total=int(len_train_loader / args.batch_size), desc="Processing test images"):

            # Get into pytorch format
            imgs, target = engine.prepare_data(imgs=imgs, target=target, args=args, device=device, disentangle_channels=disentangle_channels)
            imgs.requires_grad = True
            output, states, gates = engine.model_step(model, imgs, model_name=args.model, test=True)
            exc_w, gexc_w, gexc_u = [], [], []
            inh_w, ginh_w, ginh_u = [], [], []

            diff = ((output.sigmoid().flatten() - human_data) ** 2).mean()
            diff.backward()

            exc_wgrads = model.module.unit1.w_exc.grad.view(1, -1).detach().cpu().numpy()
            inh_wgrads = model.module.unit1.w_exc.grad.view(1, -1).detach().cpu().numpy()

            gexc_w_wgrads = model.module.unit1.e_w_gate.weight.grad.view(1, -1).detach().cpu().numpy()
            gexc_u_wgrads = model.module.unit1.e_u_gate.weight.grad.view(1, -1).detach().cpu().numpy()

            ginh_w_wgrads = model.module.unit1.i_w_gate.weight.grad.view(1, -1).detach().cpu().numpy()
            ginh_u_wgrads = model.module.unit1.i_u_gate.weight.grad.view(1, -1).detach().cpu().numpy()

            # Prep_gifs needs to be an integer
            img_grad = imgs.grad
            img_grad = img_grad.permute(0, 2, 1, 3, 4)
            import pdb;pdb.set_trace()
            img_grad = img_grad[:, :, 1]
            plt.imshow(np.maximum(-img_grad[5].detach().cpu(), 0).mean(0));plt.show()
            imgs = imgs.detach()
            if "hgru" in args.model:
                data_results_folder = os.path.join(results_folder, "gradient_dist_{}_speed_{}_length_{}".format(dist, speed, length))
                os.makedirs(data_results_folder, exist_ok=True)
                engine.plot_results(states, imgs, target, output=output, timesteps=timesteps, gates=img_grad, prep_gifs=prep_gifs, results_folder=data_results_folder, grads_passed=True)


if __name__ == '__main__':
    length = args.length
    speed = args.speed
    dist = args.dist
    # perfs = np.load(os.path.join(directory, "val.npz"))["loss"]
    # arg_perf = np.argmin(perfs)
    res_dir = "{}_{}_{}".format(length, speed, dist)
    results_folder = os.path.join('results', res_dir, args.name)
    if args.ckpt is None:
        eval_best_model(directory=results_folder, model=args.model)
    else:
        evaluate_model(results_folder=results_folder, args=args)

