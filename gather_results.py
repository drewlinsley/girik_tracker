import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from glob import glob


directories = {
    "hgru_v2": 105027,
    "hgru_wider_32": 107254,
    "r3d_1e-2": 33166785,
    "r3d_1e-3": 33166785,
    "r3d_1e-4": 33166785,
    "mc3_1e-4": 11490753,
    "mc3_1e-3": 11490753,
    "mc3_1e-2": 11490753,
    "r2plus_1e-4": 31300638,
    "r2plus_1e-3": 31300638,
    "r2plus_1e-2": 31300638,
    "imagenet_r3d_1e-2": 33166785,
    "imagenet_r3d_1e-3": 33166785,
    "imagenet_r2plus_1e-2": 31300638,
    "imagenet_r3d_1e-4": 33166785,
    "imagenet_r2plus1_1e-4": 31300638,
    "imagenet_r2plus1_1e-3": 31300638,
    "imagenet_mc3_1e-2": 11490753,
    "imagenet_mc3_1e-3": 11490753,
    "imagenet_mc3_1e-4": 11490753,
    "nostride_r3d_1e-3_fac_4": 2123169,
    "nostride_r3d_1e-4_fac_4": 2123169,
    "nostride_r3d_1e-5_fac_4": 2123169,
    "slowfast_1e-3": 33646793,
    "slowfast_1e-5": 33646793,
}

remap_names = {
    "hgru_v2": "hgru_no_attention"
}

perf = []
for k, v in directories.items():
    res_files = glob(os.path.join("results", k, "test_perf*.npz"))
    for r in res_files:
        data = np.load(r)
        splits = r.split("_")
        distractors = int(splits[-5])
        speed = int(splits[-3])
        length = int(splits[-1].split(".")[0])
        lr_trimmed_model = k.split("_1")[0]
        try:
            perf.append([lr_trimmed_model, data["arr_0"], data["arr_1"], v, distractors, speed, length])
        except:
            print("Failed to load {}".format(k))
        del data.f
        data.close()
df = pd.DataFrame(np.stack(perf, 0), columns=["model", "accuracy", "loss", "parameters", "distractors", "speed", "length"])
df.accuracy = pd.to_numeric(df.accuracy)
df.loss = pd.to_numeric(df.loss)
df.parameters = pd.to_numeric(df.parameters)
df.distractors = pd.to_numeric(df.distractors)
df.speed = pd.to_numeric(df.speed)
df.length = pd.to_numeric(df.length)

distractor_df = df[np.logical_and(df.speed==1, df.length==64)]
distractor_df = distractor_df.groupby(["model", "distractors"]).max()
distractor_df = distractor_df.reset_index()
distractor_df["log_parameters"] = np.log2(distractor_df.parameters)
distractor_df = distractor_df[distractor_df.model!="imagenet_r2plus"]
# g = sns.FacetGrid(distractor_df, col="distractors")
# g.map_dataframe(sns.scatterplot, x="parameters", y="accuracy", hue="parameters")
# g.set_axis_labels("Number of parameters", "Accuracy")
# plt.show()

# sns.set_theme(style="white")
# sns.set_theme()
fig = plt.figure()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
# sns.set_context("paper")
sns.set_context("talk")
# sns.set_context("poster")
sns.relplot(data=distractor_df, x="distractors", y="accuracy", hue="model", size="log_parameters", alpha=.75, palette="colorblind", height=6, sizes=(40, 700))
plt.show()
plt.close(fig)



distractor_df = df[np.logical_and(df.speed==1, df.distractors==14)]
distractor_df = distractor_df.groupby(["model", "length"]).max()
distractor_df = distractor_df.reset_index()
distractor_df["log_parameters"] = np.log2(distractor_df.parameters)
distractor_df = distractor_df[distractor_df.model!="imagenet_r2plus"]
# g = sns.FacetGrid(distractor_df, col="distractors")
# g.map_dataframe(sns.scatterplot, x="parameters", y="accuracy", hue="parameters")
# g.set_axis_labels("Number of parameters", "Accuracy")
# plt.show()

# sns.set_theme(style="white")
# sns.set_theme()
fig = plt.figure()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
# sns.set_context("paper")
sns.set_context("talk")
# sns.set_context("poster")
sns.relplot(data=distractor_df, x="length", y="accuracy", hue="model", size="log_parameters", alpha=.75, palette="colorblind", height=6, sizes=(40, 700))
plt.show()
plt.close(fig)


