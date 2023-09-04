import numpy as np
import pandas as pd
import nibabel
import json
import shutil
import os
import h5py
import subprocess
import tqdm

def save_segmentation(series_folder: str, out_file: str, convert_labels=True):
    # execute total_segmentator_execute.sh script with the series_folder as input
    subprocess.run(["bash", "total_segmentator_execute.sh", series_folder])
    # load the segmentation from the temporary file and delete it
    segmentations = np.array(nibabel.load("total_segmentator_TMP/temp.nii").get_fdata()).astype(np.uint8)
    os.remove("total_segmentator_TMP/temp.nii")
    if convert_labels:
        segmentations = (segmentations == 5).astype(np.uint8) + (segmentations == 1).astype(np.uint8) * 2\
                    + (segmentations == 3).astype(np.uint8) * 3 + (segmentations == 2).astype(np.uint8) * 4 + ((segmentations >= 55) & (segmentations <= 57)).astype(np.uint8) * 5

    with h5py.File(out_file, "w") as f:
        f.create_dataset("segmentation_labels", data=segmentations, dtype=np.uint8, compression="gzip", compression_opts=0)

if __name__ == "__main__":
    segmentator_results_folder = "total_segmentator_results"
    if not os.path.isdir(segmentator_results_folder):
        os.mkdir(segmentator_results_folder)

    series_meta = pd.read_csv("data/train_series_meta.csv", index_col=1)
    patient_injuries = pd.read_csv("data/train.csv", index_col=0)

    patient_with_injuries = list(patient_injuries.loc[patient_injuries["any_injury"] == 1].index)

    count = 0
    for patient_id in patient_with_injuries:
        patient_folder = os.path.join("data", "train_images", str(patient_id))
        for series_id in os.listdir(patient_folder):
            series_folder = os.path.join(patient_folder, series_id)
            outfile = os.path.join(segmentator_results_folder, series_id + ".hdf5")

            if os.path.isfile(os.path.join("data", "segmentations", series_id + ".nii")):
                continue

            if os.path.isfile(outfile):
                continue
            
            save_segmentation(series_folder, outfile, convert_labels=False)

            count += 1
            if count % 100 == 0:
                print("-----------------------------------------------------------------------------------------")
                print("Processed {} series".format(count))
                print("-----------------------------------------------------------------------------------------")