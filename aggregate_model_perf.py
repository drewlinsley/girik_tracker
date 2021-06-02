import os
from glob import glob
import numpy as np
import pandas as pd


glob_dirs = [
    '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results',
    '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results_new',
    '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/pathfinder_experiments-master/results',
]
keep_models = [
    "imagenet_mc3",
    "imagenet_m3d",
    "m3d",
    "imagenet_r2plus1",
    "imagenet_r3d",
    "mc3",
    "r3d",
    "r2plus",
    "nostride_r3d",
    "nostride_video_cc_small",
    "gru",
    "timesformer_facebook",
    "timesformer_facebook_in",
    "in_timesformer_facebook_in",
    "space_in_timesformer_facebook",
    "hgru_soft",
    "hgru_TEST",
    "hgru_wider_32"
]

exp_dict = {}
for d in glob_dirs:
    exp_files = glob(os.path.join(d, "*"))
    for e in exp_files:
        exp = e.split(os.path.sep)[-1]
        model_dict = {}
        model_files = glob(os.path.join(e, "*val.npz"))
        for m in model_files:
            model_name = m.split(os.path.sep)[-1]
            model_name = model_name.split("val")[0]
            try:
                perf = np.load(m)["balacc"].max()  # noqa
            except:  # noqa
                print("Failed to load {}".format(m))
                perf = None
            if model_name in model_dict and perf is not None:
                if perf > model_dict[model_name][1]:
                    model_dict[model_name] = [m.split("val")[0], perf]
            elif perf is not None:
                model_dict[model_name] = [m.split("val")[0], perf]

        # The old way of saving in folders.
        alt_model_files = glob(os.path.join(e, "*"))
        for m in alt_model_files:
            if os.path.isdir(m):
                val_file = glob(os.path.join(m, "*val.npz"))
                if len(val_file):
                    val_file = val_file[0]
                    model_name = m.split(os.path.sep)[-1]
                    try:
                        perf = np.load(val_file)["balacc"].max()  # noqa
                    except:  # noqa
                        print("Failed to load {}".format(val_file))
                        perf = None
                    if model_name in model_dict and perf is not None:
                        if perf > model_dict[model_name][1]:
                            model_dict[model_name] = [m, perf]
                    elif perf is not None:
                        model_dict[model_name] = [m, perf]
        if exp in exp_dict:
            exp_dict[exp].update(model_dict)
        else:
            exp_dict[exp] = model_dict

# Make the hgru dict
hgru_keys = [
    "hgru_wider_32",
    "ffhgru_no_add",
    "ffhgru_no_inh",
    "ffhgru_mult_add",
    "ffhgru_only_add",
    "ffhgru_tanh",
    "ffhgru_no_mult"
]

# Loop through exp_dict['64_1_14'].items() for each key
mac_paths, hgru_paths = {}, {}
for hgru in hgru_keys:
    perf = 0
    for k, v in exp_dict['64_1_14'].items():
        if hgru in k and v[1] > perf:
            path, perf = v
            mac_path = path.replace("/media/data_cifs/", "/cifs/data/tserre/CLPS_Serre_Lab/")  # noqa
            mac_paths[hgru] = [mac_path, perf]
            hgru_paths[hgru] = [path, perf]
df = pd.DataFrame.from_dict(mac_paths)
df.to_pickle("hgru_model_info.pkl")

# Make the mturk run script
exp_cmds = [
    "CUDA_VISIBLE_DEVICES=0 python generate_mturk_preds.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name {} --parallel --length=64 --speed=1 --dist=14 --set_name=no_gen --b=72\n",
    "CUDA_VISIBLE_DEVICES=0 python generate_mturk_preds.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name {} --parallel --length=64 --speed=1 --dist=14 --set_name=gen_1_25_64 --b=40\n",
    "CUDA_VISIBLE_DEVICES=0 python generate_mturk_preds.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name {} --parallel --length=64 --speed=1 --dist=14 --set_name=gen_1_14_128 --b=40\n"
]
paths = df.iloc[0].to_numpy()
perfs = df.iloc[1].to_numpy()
model_remap = {
    "bak_hgru_wider_32": "ffhgru",
    "hgru_wider_32": "ffhgru",
}
mturk_cmds = []
for exp in exp_cmds:
    for path in paths:
        name = path.split(os.path.sep)[-1]
        model = name.split("_1e")[0]
        if model in model_remap:
            model = model_remap[model]
        cmd = exp.format(model, path)
        mturk_cmds.append(cmd)
outname = "aggregated_turk.sh"
with open(outname, 'w') as out_file:
    out_file.writelines(mturk_cmds)

# Make the test script
ref_exps = {
    '32_1_14': [32, 1, 14, "CUDA_VISIBLE_DEVICES={} python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name {} --parallel --length=32 --speed=1 --dist=14 --which_tests=32\n"],
    '_32_1_14': [32, 1, 14, "CUDA_VISIBLE_DEVICES={} python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name {} --parallel --length=32 --speed=1 --dist=14 --which_tests=of32\n"],
    '64_1_14': [64, 1, 14, "CUDA_VISIBLE_DEVICES={} python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name {} --parallel --length=64 --speed=1 --dist=14 --which_tests=64\n"],
    '_64_1_14': [64, 1, 14, "CUDA_VISIBLE_DEVICES={} python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name {} --parallel --length=64 --speed=1 --dist=14 --which_tests=of64\n"]
}
GPU = 1
# exp_cmds = [
#     "CUDA_VISIBLE_DEVICES={} python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name {} --parallel --length=32 --speed=1 --dist=14 --which_tests=32\n",
#     "CUDA_VISIBLE_DEVICES={} python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name _{} --parallel --length=32 --speed=1 --dist=14 --which_tests=32\n",
#     "CUDA_VISIBLE_DEVICES={} python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name {} --parallel --length=64 --speed=1 --dist=14 --which_tests=64\n",
#     "CUDA_VISIBLE_DEVICES={} python test_model.py --print-freq 20 --lr 1e-03 --epochs 300 --model {} --name _{} --parallel --length=64 --speed=1 --dist=14 --which_tests=64\n",
# ]
for re, ve in ref_exps.items():
    exp_list = []
    length, speed, dist, exp_cmd = ve
    if re not in exp_dict:
        continue
    exp_models = exp_dict[re]
    for k, v in exp_models.items():
        path = v[0]
        name = path.split(os.path.sep)[-1]
        model = name.split("_1e")[0]
        model = model.split("_3e")[0]
        if "imagenet_" in model:
            model = model.split("imagenet_")[1]
        if model in model_remap:
            model = model_remap[model]
        it_exp_cmd = exp_cmd.format(GPU, model, name, length, speed, dist)
        print(model)
        for m in keep_models:
            if model == "ffhgru_no_inh" or model == "ffhgru_no_add" or model == "ffhgru_no_mult" or model == "ffhgru_mult_add" or model == "ffhgru_only_add" or model == "ffhgru_tanh" or model == "ffhgru_no_attention" or model == "ffhgru_only_add":
                pass
            elif m in model:
                exp_list.append(it_exp_cmd)
                break
    outname = "test_all_models_{}.sh".format(re)
    with open(outname, 'w') as out_file:
        out_file.writelines(exp_list)

