import os
import numpy as np
import pydicom
import tqdm

import scan_preprocessing

ct_folder = os.path.join("data", "train_images")
np_folder = "data_npy"

def get_array_and_zpos(series_folder: str) -> (np.ndarray, np.ndarray):
    ct_scan_files = [int(dcm[:-4]) for dcm in os.listdir(series_folder)]
    ct_scan_files.sort()

    min_slice = ct_scan_files[0]
    max_slice = ct_scan_files[-1]

    # Load the data
    ct_3D_image = None
    z_positions = np.zeros((max_slice - min_slice + 1,), dtype=np.float32)
    for slice_number in range(min_slice, max_slice + 1):
        dcm_file = os.path.join(series_folder, "{}.dcm".format(slice_number))
        dcm_data = pydicom.dcmread(dcm_file)
        slice_array = scan_preprocessing.to_float_array(dcm_data)
        if ct_3D_image is None:
            ct_3D_image = np.zeros(
                (max_slice - min_slice + 1, slice_array.shape[0], slice_array.shape[1]),
                dtype=np.float32)
        ct_3D_image[slice_number - min_slice] = slice_array
        z_positions[slice_number - min_slice] = dcm_data[(0x20, 0x32)].value[-1]

    return ct_3D_image, z_positions

if __name__ == "__main__":
    for patient_id in tqdm.tqdm(os.listdir(ct_folder)):
        patient_folder = os.path.join(ct_folder, patient_id)
        for series_id in os.listdir(patient_folder):
            series_folder = os.path.join(ct_folder, patient_id, series_id)
            ct_3D_image, z_positions = get_array_and_zpos(series_folder)

            npy_folder = os.path.join(np_folder, patient_id, series_id)

            if not os.path.exists(npy_folder):
                os.makedirs(npy_folder)

            np.save(os.path.join(npy_folder, "ct_3D_image.npy"), ct_3D_image)
            np.save(os.path.join(npy_folder, "z_positions.npy"), z_positions)