import numpy as np
import pandas as pd
import nibabel
import json
import shutil
import os
import h5py
import subprocess
import tqdm
import argparse
import pandas as pd

segmentator_results_folder = "total_segmentator_results"
temp_series_folder: str=None
temp_file_name: str=None
logging_folder = "total_segmentator_logging"

def save_segmentation(series_folder: str, out_file: str, convert_labels=True):
    # execute total_segmentator_execute.sh script with the series_folder as input
    subprocess.run(["bash", "total_segmentator_execute.sh", series_folder, temp_file_name])
    # load the segmentation from the temporary file and delete it
    segmentations = np.array(nibabel.load(temp_file_name + ".nii").get_fdata()).astype(np.uint8)
    os.remove(temp_file_name + ".nii") # cleanup temp nii file
    if convert_labels:
        segmentations = (segmentations == 5).astype(np.uint8) + (segmentations == 1).astype(np.uint8) * 2\
                    + (segmentations == 3).astype(np.uint8) * 3 + (segmentations == 2).astype(np.uint8) * 4 + ((segmentations >= 55) & (segmentations <= 57)).astype(np.uint8) * 5

    with h5py.File(out_file, "w") as f:
        f.create_dataset("segmentation_labels", data=segmentations, dtype=np.uint8, compression="gzip", compression_opts=0)

def save_inverted_segmentation(series_folder: str, out_file: str, convert_labels=True):
    if not os.path.isfile(temp_series_folder):
        os.mkdir(temp_series_folder)

    series_ids = [int(x[:-4]) for x in os.listdir(series_folder)]
    series_min = min(series_ids)
    series_max = max(series_ids)

    for k in range(series_min, series_max + 1):
        shutil.copy(os.path.join(series_folder, str(k) + ".dcm"),
                    os.path.join(temp_series_folder, str(series_max + series_min - k) + ".dcm"))

    # execute total_segmentator_execute.sh script with the series_folder as input
    subprocess.run(["bash", "total_segmentator_execute.sh", temp_series_folder, temp_file_name])
    # load the segmentation from the temporary file and delete it
    segmentations = np.array(nibabel.load(temp_file_name + ".nii").get_fdata()).astype(np.uint8)
    segmentations = segmentations[:, :, ::-1].copy()
    os.remove(temp_file_name + ".nii") # cleanup temp nii file
    if convert_labels:
        segmentations = (segmentations == 5).astype(np.uint8) + (segmentations == 1).astype(np.uint8) * 2 \
                        + (segmentations == 3).astype(np.uint8) * 3 + (segmentations == 2).astype(np.uint8) * 4 + (
                                    (segmentations >= 55) & (segmentations <= 57)).astype(np.uint8) * 5

    with h5py.File(out_file, "w") as f:
        f.create_dataset("segmentation_labels", data=segmentations, dtype=np.uint8, compression="gzip",
                         compression_opts=0)

    # cleanup temp series folder
    for k in range(series_min, series_max + 1):
        os.remove(os.path.join(temp_series_folder, str(k) + ".dcm"))

if __name__ == "__main__":
    # add argument stride, partition and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--partition", type=int, default=None)
    parser.add_argument("--logging", action="store_true")
    args = parser.parse_args()
    stride = args.stride
    partition = args.partition
    logging = args.logging

    assert (partition is None) == (stride == 1), "Stride must be 1 iff partition is not None"

    if partition is None:
        temp_series_folder = "total_segmentator_TMP/temp_series"
        temp_file_name = "total_segmentator_TMP/temp"
    else:
        temp_series_folder = "total_segmentator_TMP/temp_series" + str(partition)
        temp_file_name = "total_segmentator_TMP/temp" + str(partition)

    if not os.path.isdir(segmentator_results_folder):
        os.mkdir(segmentator_results_folder)
    if logging and (not os.path.isdir(logging_folder)):
        os.mkdir(logging_folder)

    series_meta = pd.read_csv("data/train_series_meta.csv", index_col=1)
    patient_injuries = pd.read_csv("data/train.csv", index_col=0)
    shape_info = pd.read_csv("data_hdf5_cropped/shape_info.csv", index_col=1)

    patient_with_injuries = list(patient_injuries.loc[patient_injuries["any_injury"] == 1].index)

    count = 0
    processed_series = []
    for patient_id in patient_with_injuries:
        patient_folder = os.path.join("data", "train_images", str(patient_id))
        for series_id in os.listdir(patient_folder):
            series_folder = os.path.join(patient_folder, series_id)
            outfile = os.path.join(segmentator_results_folder, series_id + ".hdf5")

            if (partition is None) or (count % stride == partition): # process only in the partition
                if (not os.path.isfile(os.path.join("data", "segmentations", series_id + ".nii")))\
                    and (os.path.isfile(outfile)): # skip the file if it is already processed
                    if shape_info.loc[int(series_id)]["mean_slope"] < 0:
                        save_segmentation(series_folder, outfile, convert_labels=False)
                    else:
                        save_inverted_segmentation(series_folder, outfile, convert_labels=False)

                    processed_series.append(series_id)
                    if logging:
                        if not os.path.isfile(os.path.join(logging_folder, "log_" + str(partition) + ".txt")):
                            with open(os.path.join(logging_folder, "log_" + str(partition) + ".txt"), "w") as f:
                                f.write(str(series_id) + "\n")
                        else:
                            with open(os.path.join(logging_folder, "log_" + str(partition) + ".txt"), "a") as f:
                                f.write(str(series_id) + "\n")

            count += 1
            if count % 100 == 0:
                print("-----------------------------------------------------------------------------------------")
                print("Processed {} series".format(count))
                print("-----------------------------------------------------------------------------------------")