import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

shapes = pd.read_csv("data_npy/shape_info.csv")
# plot the histogram along with kde of the shape_h column
sns.displot(shapes["shape_h"], kde=True)
plt.title("Maximum height: {}".format(shapes["shape_h"].max()))
plt.show()
# plot the histogram along with kde of the shape_w column
sns.displot(shapes["shape_w"], kde=True)
plt.title("Maximum width: {}".format(shapes["shape_w"].max()))
plt.show()

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

# save the list of not equally spaced series
with open("not_equally_spaced.txt", "w") as f:
    for k in not_equally_spaced:
        f.write("{} {}\n".format(shapes.iloc[k]["patient_id"], shapes.iloc[k]["series_id"]))

print("Asserting...")
folder = "data/train_images"
for patient_id in os.listdir(folder):
    patient_folder = os.path.join(folder, patient_id)
    for series_id in os.listdir(patient_folder):
        assert os.path.exists(os.path.join("data_hdf5", patient_id, series_id, "ct_3D_image.hdf5"))

print("Success.")