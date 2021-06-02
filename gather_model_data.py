import os
from glob import glob
import numpy as np
import pandas as pd


glob_dirs = [
    '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/cifs_results',
    '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/pathfinder_experiments-master/results',
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
                model_dict[model_name][1] = max(model_dict[model_name][1], perf)
            elif perf is not None:
                model_dict[model_name] = [m, perf]

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
                        model_dict[model_name][1] = max(model_dict[model_name][1], perf)  # noqa
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
hgru_paths = {}
for hgru in hgru_keys:
    perf = 0
    for k, v in exp_dict['64_1_14'].items():
        if hgru in k and v[1] > perf:
            path, perf = v
            path = path.replace("/media/data_cifs/", "/cifs/data/tserre/CLPS_Serre_Lab/")
            hgru_paths[hgru] = [path, perf]
df = pd.DataFrame.from_dict(hgru_paths)
df.to_csv("hgru_model_info.csv")

