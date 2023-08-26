import os
import numpy as np
import pydicom
import tqdm
import cv2
import pandas as pd
import torch

import convert_to_npy

ct_folder = os.path.join("data", "train_images")
cropped_hdf_folder = "data_hdf5_cropped"

def crop_3d_volume(ct_3D_image):
    image_locs = torch.tensor(ct_3D_image, dtype=torch.float32, device="cuda")
    image_locs = image_locs > 1e-6  # shape (depth, height, width)

    heights = torch.any(torch.any(image_locs, dim=0), dim=-1).cpu().numpy()  # shape (height,)
    widths = torch.any(torch.any(image_locs, dim=0), dim=-2).cpu().numpy()  # shape (width,)

    heights = np.argwhere(heights)
    widths = np.argwhere(widths)

    return ct_3D_image[:, heights.min():(heights.max() + 1), widths.min():(widths.max() + 1)]


if __name__ == "__main__":
    shape_info = {"patient_id": [], "series_id": [], "shape_h": [], "shape_w": [], "z_positions": []}
    if not os.path.isdir(cropped_hdf_folder):
        os.mkdir(cropped_hdf_folder)

    for patient_id in tqdm.tqdm(os.listdir(ct_folder)):
        patient_folder = os.path.join(ct_folder, patient_id)
        for series_id in os.listdir(patient_folder):
            series_folder = os.path.join(ct_folder, patient_id, series_id)
            ct_3D_image, z_positions = convert_to_npy.get_array_and_zpos(series_folder) # convert dicom series to numpy array
            ct_3D_image = crop_3d_volume(ct_3D_image) # crop the 3D volume to the smallest possible size


            shape = (ct_3D_image.shape[1], ct_3D_image.shape[2])

            hdf_folder = os.path.join(cropped_hdf_folder, patient_id, series_id)

            if not os.path.exists(hdf_folder):
                os.makedirs(hdf_folder)

            np.save(os.path.join(hdf_folder, "ct_3D_image.npy"), ct_3D_image)
            np.save(os.path.join(hdf_folder, "z_positions.npy"), z_positions)

            shape_info["patient_id"].append(patient_id)
            shape_info["series_id"].append(series_id)
            shape_info["shape_h"].append(shape[0])
            shape_info["shape_w"].append(shape[1])
            shape_info["z_positions"].append(str(list(z_positions)))

    shape_info = pd.DataFrame(shape_info)
    shape_info.to_csv("data_hdf_cropped/shape_info.csv", index=False)
