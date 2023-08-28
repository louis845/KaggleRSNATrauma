import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

shapes = pd.read_csv("data_npy/shape_info.csv")

slopes = []
not_equally_spaced = []
for k in range(len(shapes)):
    shape = shapes.iloc[k]["z_positions"]
    z_poses = np.array([float(z_pos) for z_pos in shape[1:-1].split(", ")])

    if np.max(np.abs(np.diff(z_poses, n=2))) >= 1e-6:
        not_equally_spaced.append(k)

    slope = np.mean(np.diff(z_poses))
    slopes.append(slope)

# plot the histogram along with kde of the slopes
sns.displot(slopes, kde=True)
plt.show()

shapes["slopes"] = slopes

# get list of series with segmentations
segmentations = os.listdir("data/segmentations")
segmentations = [int(segmentation[:-4]) for segmentation in segmentations]
shape_segmentations = shapes.loc[shapes["series_id"].isin(segmentations)]
print(shape_segmentations)

# save to csv
shape_segmentations = shape_segmentations.drop(columns=["z_positions"])
shape_segmentations.to_csv("data_npy/shape_info_segmentations.csv", index=False)

print("Num positive: " + str((shape_segmentations["slopes"] > 0.0).value_counts()))
print("Num negative: " + str((shape_segmentations["slopes"] <= 0.0).value_counts()))