import os
import numpy as np
import pydicom
import tqdm
import cv2
import pandas as pd
import torch
import h5py
import nibabel

import convert_to_npy
import segmentation_processing

ct_folder = os.path.join("data", "train_images")
seg_folder = os.path.join("data", "segmentations")
cropped_hdf_folder = "data_hdf5_cropped"
cropped_segmentation_folder = "data_segmentation_hdf_cropped"

def crop_3d_volume(ct_3D_image, segmentation_arr):
    image_locs = torch.tensor(ct_3D_image, dtype=torch.float32, device="cuda")
    image_locs = image_locs > 1e-6  # shape (depth, height, width)

    heights = torch.any(torch.any(image_locs, dim=0), dim=-1).cpu().numpy()  # shape (height,)
    widths = torch.any(torch.any(image_locs, dim=0), dim=-2).cpu().numpy()  # shape (width,)

    heights = np.argwhere(heights)
    widths = np.argwhere(widths)

    if segmentation_arr is None:
        return ct_3D_image[:, heights.min():(heights.max() + 1), widths.min():(widths.max() + 1)], None
    else:
        return ct_3D_image[:, heights.min():(heights.max() + 1), widths.min():(widths.max() + 1)],\
            segmentation_arr[:, heights.min():(heights.max() + 1), widths.min():(widths.max() + 1), :]


if __name__ == "__main__":
    shape_info = {"patient_id": [], "series_id": [], "shape_h": [], "shape_w": [], "z_positions": [], "mean_slope": []}
    if not os.path.isdir(cropped_hdf_folder):
        os.mkdir(cropped_hdf_folder)
    if not os.path.isdir(cropped_segmentation_folder):
        os.mkdir(cropped_segmentation_folder)

    for patient_id in tqdm.tqdm(os.listdir(ct_folder)):
        patient_folder = os.path.join(ct_folder, patient_id)
        for series_id in os.listdir(patient_folder):
            series_folder = os.path.join(ct_folder, patient_id, series_id)
            ct_3D_image, z_positions, dcm_data = convert_to_npy.get_array_and_zpos(series_folder) # convert dicom series to numpy array

            # load segmentation
            segmentation_file = os.path.join(seg_folder, series_id + ".nii")
            if os.path.isfile(segmentation_file):
                segmentation_arr = np.array(nibabel.load(segmentation_file).get_fdata())
                segmentation_arr = segmentation_processing.to_class_array(dcm_data, segmentation_arr)
            else:
                segmentation_arr = None

            ct_3D_image, segmentation_arr = crop_3d_volume(ct_3D_image, segmentation_arr) # crop the 3D volume to the smallest possible size

            shape = (ct_3D_image.shape[1], ct_3D_image.shape[2])

            hdf_folder = os.path.join(cropped_hdf_folder, patient_id, series_id)

            if not os.path.exists(hdf_folder):
                os.makedirs(hdf_folder)

            with h5py.File(os.path.join(hdf_folder, "ct_3D_image.hdf5"), "w") as f:
                f.create_dataset("ct_3D_image", data=ct_3D_image, dtype=np.float16, compression="gzip", compression_opts=0)
            if segmentation_arr is not None:
                assert segmentation_arr.shape[0] == ct_3D_image.shape[0]
                assert segmentation_arr.shape[1] == ct_3D_image.shape[1]
                assert segmentation_arr.shape[2] == ct_3D_image.shape[2]
                with h5py.File(os.path.join(cropped_segmentation_folder, "{}.hdf5".format(series_id)), "w") as f:
                    f.create_dataset("segmentation_arr", data=segmentation_arr, dtype=np.uint8, compression="gzip", compression_opts=0)
            np.save(os.path.join(hdf_folder, "z_positions.npy"), z_positions)

            shape_info["patient_id"].append(patient_id)
            shape_info["series_id"].append(series_id)
            shape_info["shape_h"].append(shape[0])
            shape_info["shape_w"].append(shape[1])
            shape_info["z_positions"].append(str(list(z_positions)))
            shape_info["mean_slope"].append(np.mean(np.diff(z_positions)))

    shape_info = pd.DataFrame(shape_info)
    shape_info.to_csv("data_hdf5_cropped/shape_info.csv", index=False)
