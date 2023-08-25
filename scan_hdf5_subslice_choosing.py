import numpy as np
import torch
import h5py
import tqdm
import pandas as pd

import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, default="data_hdf5_cropped")
parser.add_argument("--limitation", type=int, default=1000)
args = parser.parse_args()
output_folder = args.output_folder

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

count = 0
breaked = False
for patient_id in tqdm.tqdm(os.listdir("data_hdf5")):
    patient_folder = os.path.join("data_hdf5", patient_id)
    patient_output_folder = os.path.join(output_folder, patient_id)
    if not os.path.isdir(patient_folder):
        continue
    if not os.path.isdir(patient_output_folder):
        os.makedirs(patient_output_folder)
    else:
        continue
    for series_id in os.listdir(patient_folder):
        series_folder = os.path.join(patient_folder, series_id)
        series_output_folder = os.path.join(patient_output_folder, series_id)
        if not os.path.isdir(series_output_folder):
            os.makedirs(series_output_folder)

        with h5py.File(os.path.join(series_folder, "ct_3D_image.hdf5"), "r") as f:
            ct_3D_image = f["ct_3D_image"][()]

        if ct_3D_image.shape[0] > 40:
            # subsample the first axis to 40, evenly spaced
            ct_3D_image = ct_3D_image[np.linspace(0, ct_3D_image.shape[0] - 1, 40, dtype=int), ...]
            # save to new folder
            with h5py.File(os.path.join(series_output_folder, "ct_3D_image.hdf5"), "w") as f:
                f.create_dataset("ct_3D_image", data=ct_3D_image, dtype=np.float16, compression="gzip", compression_opts=0)
        else:
            shutil.copyfile(os.path.join(series_folder, "ct_3D_image.hdf5"), os.path.join(series_output_folder, "ct_3D_image.hdf5"))
        del ct_3D_image

    count += 1
    if count >= args.limitation:
        breaked = True
        break

if not breaked:
    print("Finished all patients.")
else:
    print("Finished a subsample.")
