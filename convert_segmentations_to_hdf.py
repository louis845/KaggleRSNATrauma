import segmentation_processing
import pandas as pd
import numpy as np
import pydicom
import nibabel
import h5py
import tqdm

import os

series_data = pd.read_csv("data/train_series_meta.csv", index_col=1)
series_data.index = series_data.index.astype(str)
def get_segmentation_array(segmentation_id: str) -> np.ndarray:
    patient_id = series_data.loc[segmentation_id, "patient_id"]
    series_folder = os.path.join("data", "train_images", str(patient_id), segmentation_id)
    single_slice = os.listdir(series_folder)[0]
    dcm_file = os.path.join(series_folder, single_slice)
    dcm_data = pydicom.dcmread(dcm_file)

    segmentation_file = os.path.join("data", "segmentations", segmentation_id + ".nii")
    segmentation_arr = np.array(nibabel.load(segmentation_file).get_fdata())
    segmentation_arr = segmentation_processing.to_class_array(dcm_data, segmentation_arr)

    return segmentation_arr

if __name__ == "__main__":
    if not os.path.isdir("data_segmentation_hdf"):
        os.mkdir("data_segmentation_hdf")

    for segmentation_file in tqdm.tqdm(os.listdir("data/segmentations")):
        segmentation_arr = get_segmentation_array(segmentation_file[:-4])
        with h5py.File(os.path.join("data_segmentation_hdf", segmentation_file[:-4] + ".hdf"), "w") as f:
            f.create_dataset("segmentation", data=segmentation_arr, compression="gzip", compression_opts=0, dtype=np.uint8)
