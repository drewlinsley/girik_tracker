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
import matplotlib
# import imageio
from torch._six import inf
from torchvideotransforms import video_transforms, volume_transforms


torch.backends.cudnn.benchmark = True

global best_prec1
best_prec1 = 0
args = parser.parse_args()
video_transform_list = [video_transforms.RandomHorizontalFlip(0.5), video_transforms.RandomVerticalFlip(0.5)]  # , volume_transforms.ClipToTensor(div_255=False)]
transforms = video_transforms.Compose(video_transform_list)
use_augmentations = False

pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32/14_dist/tfrecords/'
# pf_root = '/media/data_cifs/projects/prj_tracking/downsampled_constrained_red_blue_datasets_64_32_32/25_dist/tfrecords/'
# print("Loading training dataset")
# train_loader = tfr_data_loader(data_dir=pf_root+'train-*', batch_size=args.batch_size, drop_remainder=True)

print("Loading validation dataset")
# val_loader = tfr_data_loader(data_dir=pf_root+'test-*', batch_size=args.batch_size, drop_remainder=True)
val_loader = tfr_data_loader(data_dir=pf_root+'train-*', batch_size=args.batch_size, drop_remainder=True)  # DEBUG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

len_train_loader = 20000
len_val_loader = 20000


def debug_plot(img):
    from matplotlib import pyplot as plt
    import tkinter
    import matplotlib
    matplotlib.use('TkAgg')
    pl = img.cpu().numpy().transpose(0, 2, 3, 4, 1)
    plt.subplot(131);plt.imshow(pl[0, 0]);plt.subplot(132);plt.imshow(pl[0, 5]);plt.subplot(133);plt.imshow(pl[0, 10]);plt.show()


def save_npz(epoch, log_dict, results_folder, savename='train'):

    with open(results_folder + savename + '.npz', 'wb') as f:
        np.savez(f, **log_dict)


if __name__ == '__main__':
    
    results_folder = 'results/{0}/'.format(args.name)
    # os.mkdir(results_folder)
    os.makedirs(results_folder, exist_ok=True)
    
    exp_logging = args.log
    jacobian_penalty = args.penalty

    timesteps = 64
    fb_kernel_size = 7
    dimensions = 48
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
    else:
        raise NotImplementedError("Model not found.")

    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    if args.parallel is True:
        model = torch.nn.DataParallel(model).to(device)
        print("Loading parallel finished on GPU count:", torch.cuda.device_count())
    else:
        model = model.to(device)
        print("Loading finished")

    criterion = torch.nn.BCEWithLogitsLoss().to(device)

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

    lr_init = args.lr

    val_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': []}
    train_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': [], 'jvpen': [], 'scaled_loss': []}

    args.val_freq = 2000

    exp_loss = None
    scale = torch.Tensor([1.0]).to(device)
    for epoch in range(args.start_epoch, args.epochs):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        precision = AverageMeter()
        recall = AverageMeter()
        f1score = AverageMeter()

        time_since_last = time.time()
        model.eval()
        end = time.perf_counter()

        for idx, (imgs, target) in enumerate(val_loader):
            data_time.update(time.perf_counter() - end)
            
            # Get into pytorch format
            # imgs = torch.from_numpy(imgs.numpy())
            # imgs = imgs.permute(0,4,1,2,3)
            imgs = imgs.numpy()
            imgs = imgs.transpose(0,4,1,2,3)
            target = torch.from_numpy(np.vectorize(ord)(target.numpy()))
            # imgs = imgs.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            # Convert imgs to 1-channel
            # imgs = imgs.mean(1, keepdim=True)
            imgs = imgs / 255.  # Normalize to [0, 1]

            mask = imgs.sum(1).round()
            # proc_imgs = torch.zeros_like(imgs)
            # proc_imgs[:, 1] = (mask == 1).float()
            # proc_imgs[:, 2] = (mask == 2).float()
            # proc_imgs[:, 0] = (mask == 3).float()
            proc_imgs = np.zeros_like(imgs)
            proc_imgs[:, 1] = (mask == 1).astype(imgs.dtype)
            proc_imgs[:, 2] = (mask == 2).astype(imgs.dtype)
            proc_imgs[:, 0] = (mask == 3).astype(imgs.dtype)
            if use_augmentations:
                imgs = transforms(proc_imgs)
                imgs = np.stack(imgs, 0)
            else:
                imgs = proc_imgs
            imgs = torch.from_numpy(proc_imgs)            
            imgs = imgs.to(device, dtype=torch.float)

            # Run training
            output, states, gates = model.forward(imgs, testmode=True)
            states = states.detach().cpu().numpy()
            gates = gates.detach().cpu().numpy()
            rng = np.arange(0, 64, 8)
            rng = np.concatenate((np.arange(0,64,8), [63]))
            img = imgs.cpu().numpy()
            from matplotlib import pyplot as plt
            sel = target.float().reshape(-1, 1) == (output > 0).float()
            sel = sel.cpu().numpy()
            sel = np.where(sel)[0]
            sel = sel[0]
            fig = plt.figure()
            for idx, i in enumerate(rng):
                print(idx)
                plt.subplot(3, 9, idx + 1)
                plt.axis("off")
                plt.imshow(img[sel, :, i].transpose(1, 2, 0))
                plt.title("Img")
                plt.subplot(3, 9, idx + 1 + 9)
                plt.axis("off")
                plt.imshow(gates[sel, i].squeeze())
                plt.title("Attn")
                plt.subplot(3, 9, idx + 1 + 18)
                plt.title("Activity")
                plt.axis("off")
                plt.imshow(states[sel, i].squeeze())
            plt.suptitle("Batch acc: {}, Prediction: {}, Label: {}".format((target.float() == (output > 0).float()).float().mean(), output[sel].cpu(), target[sel]))
            plt.show()
            plt.close(fig)

            loss = criterion(output, target.float().reshape(-1, 1))
            losses.update(loss.data.item(), 1)
            prec1, preci, rec, f1s = acc_scores(target[:], output.data[:])
            
            top1.update(prec1.item(), 1)
            precision.update(preci.item(), 1)
            recall.update(rec.item(), 1)
            f1score.update(f1s.item(), 1)
            batch_time.update(time.perf_counter() - end)
            
            end = time.perf_counter()
            if idx % (args.print_freq) == 0:
                time_now = time.time()
                print_string = 'Epoch: [{0}][{1}/{2}]  lr: {lr:g}  Time: {batch_time.val:.3f} (itavg:{timeiteravg:.3f}) '\
                               '({batch_time.avg:.3f})  Data: {data_time.val:.3f} ({data_time.avg:.3f}) ' \
                               'Loss: {loss.val:.8f} ({lossprint:.8f}) ({loss.avg:.8f})  bal_acc: {top1.val:.5f} '\
                               '({top1.avg:.5f}) preci: {preci.val:.5f} ({preci.avg:.5f}) rec: {rec.val:.5f} '\
                               '({rec.avg:.5f})  f1: {f1s.val:.5f} ({f1s.avg:.5f}) jvpen: {jpena:.12f} {timeprint:.3f} losscale:{losscale:.5f}'\
                               .format(epoch, idx, len_val_loader, batch_time=batch_time, data_time=data_time, loss=losses,
                                       lossprint=mean(losses.history[-args.print_freq:]), lr=0,
                                       top1=top1, timeiteravg=mean(batch_time.history[-args.print_freq:]),
                                       timeprint=time_now - time_since_last, preci=precision, rec=recall,
                                       f1s=f1score, jpena=0., losscale=scale.item())
                print(print_string)
                time_since_last = time_now
                with open(results_folder + args.name + '.txt', 'a+') as log_file:
                    log_file.write(print_string + '\n')
        #lr_scheduler.step()
        print(epoch)
        train_log_dict['loss'].extend(losses.history)
        train_log_dict['balacc'].extend(top1.history)
        train_log_dict['precision'].extend(precision.history)
        train_log_dict['recall'].extend(recall.history)
        train_log_dict['f1score'].extend(f1score.history)

