import h5py
import numpy as np
import os
import shutil
import tqdm

if not os.path.isdir("data_hdf5"):
    os.mkdir("data_hdf5")

for patient_id in tqdm.tqdm(os.listdir("data_npy")):
    patient_folder = os.path.join("data_npy", patient_id)
    if not os.path.isdir(patient_folder):
        continue
    for series_id in os.listdir(patient_folder):
        series_folder = os.path.join(patient_folder, series_id)

        ct_3D_image = np.load(os.path.join(series_folder, "ct_3D_image.npy"))

        hdf5_folder = os.path.join("data_hdf5", patient_id, series_id)
        if not os.path.exists(hdf5_folder):
            os.makedirs(hdf5_folder)

        with h5py.File(os.path.join(hdf5_folder, "ct_3D_image.hdf5"), "w") as f:
            f.create_dataset("ct_3D_image", data=ct_3D_image, dtype=np.float16, compression="gzip", compression_opts=0)

        # move z_positions.npy
        shutil.copyfile(os.path.join(series_folder, "z_positions.npy"), os.path.join(hdf5_folder, "z_positions.npy"))

        # remove npy folder
        shutil.rmtree(series_folder)

    # remove patient folder
    shutil.rmtree(patient_folder)
