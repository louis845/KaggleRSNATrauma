# convert the result from total segmentator to cropped hdf5

import os
import numpy as np
import pydicom
import tqdm
import cv2
import pandas as pd
import torch
import h5py

import convert_to_npy
import segmentation_processing
import convert_to_hdf_cropped

ct_folder = os.path.join("data", "train_images")
seg_folder = "total_segmentator_results"
cropped_segmentation_folder = "total_segmentator_hdf_cropped"

def main_func(stride=1, partition=0):
    patient_id_table = pd.read_csv("data/train_series_meta.csv", index_col=1)

    if not os.path.isdir(cropped_segmentation_folder):
        os.mkdir(cropped_segmentation_folder)

    count = -1
    for series_file in tqdm.tqdm(os.listdir(seg_folder)):
        count += 1

        series_id = series_file[:-5]
        file_path = os.path.join(seg_folder, series_file)

        output_file = os.path.join(cropped_segmentation_folder, series_id + ".hdf5")
        if os.path.isfile(output_file):  # if already processed, skip
            continue
        if (count % stride) != partition:
            continue
        # laod segmentation
        with h5py.File(file_path, "r") as f:
            segmentation_labels = f["segmentation_labels"][()]

        patient_id = patient_id_table.loc[int(series_id), "patient_id"]
        series_folder = os.path.join("data", "train_images", str(patient_id), series_id)
        ct_3D_image, z_positions, dcm_data = convert_to_npy.get_array_and_zpos(
            series_folder)  # convert dicom series to numpy array

        segmentation_labels = segmentation_processing.translate_labels(segmentation_labels)
        segmentation_arr = segmentation_processing.to_class_array(dcm_data, segmentation_labels)

        _, segmentation_arr = convert_to_hdf_cropped.crop_3d_volume(ct_3D_image,
                                                                    segmentation_arr)  # crop the 3D volume to the smallest possible size

        with h5py.File(os.path.join(cropped_segmentation_folder, "{}.hdf5".format(series_id)), "w") as f:
            f.create_dataset("segmentation_arr", data=segmentation_arr, dtype=np.uint8, compression="gzip",
                             compression_opts=0)

if __name__ == "__main__":
    main_func()
