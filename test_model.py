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


torch.backends.cudnn.benchmark = True
TORCHVISION = ['r3d', 'mc3', 'r2plus1', 'nostride_r3d']

global best_prec1
best_prec1 = 0
args = parser.parse_args()
video_transform_list = [video_transforms.RandomHorizontalFlip(0.5), video_transforms.RandomVerticalFlip(0.5)]  # , volume_transforms.ClipToTensor(div_255=False)]
transforms = video_transforms.Compose(video_transform_list)
use_augmentations = False
disentangle_channels = False
plot_incremental = False
debug_data = False

# pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32/14_dist/tfrecords/'
pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/14_dist/tfrecords/'
pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_128_32_32_separate_channels/14_dist/tfrecords/'
# pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_32_32_32_separate_channels/14_dist/tfrecords/'
# pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/25_dist/tfrecords/'
# pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels_skip_param_2/14_dist/tfrecords/'
# pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/0_dist/tfrecords/'
# pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels/5_dist/tfrecords/'
# pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32_separate_channels_skip_param_4/14_dist/tfrecords/'

timesteps = 128
print("Loading training dataset")
train_loader = tfr_data_loader(data_dir=os.path.join(pf_root,'train-*'), batch_size=args.batch_size, drop_remainder=True, timesteps=timesteps)

print("Loading validation dataset")
val_loader = tfr_data_loader(data_dir=os.path.join(pf_root, 'test-*'), batch_size=args.batch_size, drop_remainder=True, timesteps=timesteps)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

len_train_loader = 20000
len_val_loader = 20000


if __name__ == '__main__':
    
    results_folder = 'results/{0}/'.format(args.name)
    # os.mkdir(results_folder)
    os.makedirs(results_folder, exist_ok=True)
    
    exp_logging = args.log
    jacobian_penalty = args.penalty

    fb_kernel_size = 7
    dimensions = 32
    if args.model == 'hgru':
        print("Init model hgru ", args.algo, 'penalty: ', args.penalty)  # , 'steps: ', timesteps)
        model = models.hConvGRU(timesteps=timesteps, filt_size=15, num_iter=15, exp_name=args.name, jacobian_penalty=jacobian_penalty,
                         grad_method=args.algo)
    elif args.model == 'ffhgru':
        print("Init model ffhgru ", args.algo, 'penalty: ', args.penalty)
        model = ffhgru.FFhGRU(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'ffhgru_v2':
        print("Init model ffhgru ", args.algo, 'penalty: ', args.penalty)
        model = ffhgru.FFhGRU_v2(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'ffhgru3d':
        print("Init model ffhgru ", args.algo, 'penalty: ', args.penalty)
        model = ffhgru.FFhGRU3D(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'clock_dynamic':
        print("Init model ffhgru ", args.algo, 'penalty: ', args.penalty)
        model = ffhgru.ClockHGRU(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            clock_type="dynamic",
            grad_method='bptt')
    elif args.model == 'clock_fixed':
        print("Init model ffhgru ", args.algo, 'penalty: ', args.penalty)
        model = ffhgru.ClockHGRU(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            clock_type="fixed",
            grad_method='bptt')
    elif args.model == 'fc':
        print("Init model ffhgru ", args.algo, 'penalty: ', args.penalty)
        model = ffhgru.FC(
            dimensions=dimensions,
            timesteps=timesteps,
            kernel_size=fb_kernel_size,
            jacobian_penalty=False,
            grad_method='bptt')
    elif args.model == 'r3d':
        model = video.r3d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'nostride_r3d':
        model = nostride_video.r3d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'mc3':
        model = video.mc3_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif args.model == 'r2plus1':
        model = video.r2plus1d_18(pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    else:
        raise NotImplementedError("Model not found.")

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

    val_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': []}
    train_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': [], 'jvpen': [], 'scaled_loss': []}
    if args.pretrained:
        pre_transform = presets.VideoClassificationPresetEval((32, 32), (32, 32))

    assert args.ckpt is not None, "You must pass a checkpoint for testing."
    # model_path = "results/test_drew/saved_models/model_fscore_3325_epoch_186_checkpoint.pth.tar"
    model_path = args.ckpt
    checkpoint = torch.load(model_path)
    # Check if "module" is the first part of the key
    # check = checkpoint['state_dict'].keys()[0]
    sd = checkpoint['state_dict']
    # if "module" in check and not args.parallel:
    #     new_sd = {}
    #     for k, v in sd.items():
    #         new_sd[k.replace("module.", "")] = v
    #     sd = new_sd
    model.load_state_dict(sd)
    model.eval()
    accs = []
    losses = []
    for epoch in range(1):
        
        time_since_last = time.time()
        model.train()
        end = time.perf_counter()

        if debug_data:  # "skip" in pf_root:
            loader = train_loader
        else:
            loader = val_loader
        for idx, (imgs, target) in tqdm(enumerate(loader), total=int(len_val_loader / args.batch_size), desc="Processing test images"):
            
            # Get into pytorch format
            with torch.no_grad():
                imgs = imgs.numpy()
                imgs = imgs.transpose(0,4,1,2,3)
                target = torch.from_numpy(np.vectorize(ord)(target.numpy()))
                target = target.to(device, dtype=torch.float)

                # Convert imgs to 1-channel
                imgs = imgs / 255.  # Normalize to [0, 1]

                if disentangle_channels:
                    mask = imgs.sum(1).round()
                    proc_imgs = np.zeros_like(imgs)
                    proc_imgs[:, 1] = (mask == 1).astype(imgs.dtype)
                    proc_imgs[:, 2] = (mask == 2).astype(imgs.dtype)
                    thing_layer = (mask == 3).astype(imgs.dtype)
                    proc_imgs[:, 0] = thing_layer
                else:
                    proc_imgs = imgs
                if use_augmentations:
                    imgs = transforms(proc_imgs)
                    imgs = np.stack(imgs, 0)
                else:
                    imgs = proc_imgs
                imgs = torch.from_numpy(proc_imgs)            
                imgs = imgs.to(device, dtype=torch.float)
                if args.pretrained:
                    mu = torch.tensor([0.43216, 0.394666, 0.37645], device=device)[None, :, None, None, None]
                    stddev = torch.tensor([0.22803, 0.22145, 0.216989], device=device)[None, :, None, None, None]
                    imgs = (imgs - mu) / stddev

                # Run training
                if args.model in TORCHVISION:
                    output = model.forward(imgs)
                else:
                    output, states, gates = model.forward(imgs, testmode=True)
                loss = criterion(output, target.float().reshape(-1, 1))
                accs.append((target.reshape(-1).float() == (output.reshape(-1) > 0).float()).float().mean().cpu())
                losses.append(loss.item())
                if plot_incremental:
                    states = states.detach().cpu().numpy()
                    gates = gates.detach().cpu().numpy()
                    loss = criterion(output, target.float().reshape(-1, 1))
                    cols = (timesteps / 8) + 1
                    rng = np.arange(0, timesteps, 8)
                    rng = np.concatenate((np.arange(0,timesteps,8), [timesteps-1]))
                    img = imgs.cpu().numpy()
                    from matplotlib import pyplot as plt
                    sel = target.float().reshape(-1, 1) == (output > 0).float()
                    sel = sel.cpu().numpy()
                    sel = np.where(sel)[0]
                    sel = sel[0]
                    fig = plt.figure()
                    for idx, i in enumerate(rng):
                        print(idx)
                        plt.subplot(3, cols, idx + 1)
                        plt.axis("off")
                        plt.imshow(img[sel, :, i].transpose(1, 2, 0))
                        plt.title("Img")
                        plt.subplot(3, cols, idx + 1 + cols)
                        plt.axis("off")
                        plt.imshow((gates[sel, i].squeeze() ** 2).mean(0))
                        plt.title("Attn")
                        plt.subplot(3, cols, idx + 1 + cols + (cols - 1))
                        plt.title("Activity")
                        plt.axis("off")
                        plt.imshow(np.abs(states[sel, i].squeeze()))
                    plt.suptitle("Batch acc: {}, Prediction: {}, Label: {}".format((target.reshape(-1).float() == (output.reshape(-1) > 0).float()).float().mean(), output[sel].cpu(), target[sel]))
                    plt.show()
                    plt.close(fig)

    print("Mean accuracy: {}, mean loss: {}".format(np.mean(accs), np.mean(losses)))

    states = states.detach().cpu().numpy()
    gates = gates.detach().cpu().numpy()
    rng = np.arange(0, timesteps, 8)
    cols = (timesteps / 8) + 1
    rng = np.concatenate((np.arange(0,timesteps,8), [timesteps-1]))
    img = imgs.cpu().numpy()
    from matplotlib import pyplot as plt
    sel = target.float().reshape(-1, 1) == (output > 0).float()
    sel = sel.cpu().numpy()
    sel = np.where(sel)[0]
    sel = sel[0]
    fig = plt.figure()
    for idx, i in enumerate(rng):
        print(idx)  
        plt.subplot(3, cols, idx + 1) 
        plt.axis("off")
        plt.imshow(img[sel, :, i].transpose(1, 2, 0))
        plt.title("Img")
        plt.subplot(3, cols, idx + 1 + cols)
        plt.axis("off")
        plt.imshow((gates[sel, i].squeeze() ** 2).mean(0)) 
        plt.title("Attn")
        plt.subplot(3, cols, idx + 1 + cols + (cols - 1))
        plt.title("Activity")
        plt.axis("off")
        plt.imshow(np.abs(states[sel, i].squeeze()))
    plt.suptitle("Batch acc: {}, Prediction: {}, Label: {}".format((target.reshape(-1).float() == (output.reshape(-1) > 0).float()).float().mean(), output[sel].cpu(), target[sel]))
    plt.show()
    plt.close(fig)


