import numpy as np
import torch
import h5py
import tqdm
import pandas as pd

import os

shape_info = {"patient_id": [], "series_id": [], "shape_h": [], "shape_w": [], "area": []}
for patient_id in tqdm.tqdm(os.listdir("data_hdf5")):
    patient_folder = os.path.join("data_hdf5", patient_id)
    if not os.path.isdir(patient_folder):
        continue
    for series_id in os.listdir(patient_folder):
        series_folder = os.path.join(patient_folder, series_id)

        with h5py.File(os.path.join(series_folder, "ct_3D_image.hdf5"), "r") as f:
            ct_3D_image = f["ct_3D_image"][()]

        ct_3D_image = torch.tensor(ct_3D_image, dtype=torch.float32, device="cuda")
        ct_3D_image = ct_3D_image > 1e-6 # shape (depth, height, width)

        heights = torch.any(torch.any(ct_3D_image, dim=0), dim=-1).cpu().numpy() # shape (height,)
        widths = torch.any(torch.any(ct_3D_image, dim=0), dim=-2).cpu().numpy() # shape (width,)

        min_height = np.argwhere(heights).max() - np.argwhere(heights).min() + 1
        min_width = np.argwhere(widths).max() - np.argwhere(widths).min() + 1

        shape_info["patient_id"].append(patient_id)
        shape_info["series_id"].append(series_id)
        shape_info["shape_h"].append(min_height)
        shape_info["shape_w"].append(min_width)
        shape_info["area"].append(min_height * min_width)

# save the dataframe
shape_info = pd.DataFrame(shape_info)
shape_info.to_csv("reduced_shape.csv", index=False)
