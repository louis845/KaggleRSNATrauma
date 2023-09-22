import os
import argparse

import pandas as pd
import numpy as np
import h5py

import manager_stage1_results
import image_ROI_sampler

series_meta = pd.read_csv("data/train_series_meta.csv", index_col=1)
organs = ["liver", "spleen", "kidney"]

SAMPLING_DEPTH = 17

# create folders
extracted_folder = "EXTRACTED_STAGE1_RESULTS"
if not os.path.isdir(extracted_folder):
    os.mkdir(extracted_folder)
seg_results_fldr = os.path.join(extracted_folder, manager_stage1_results.SEGMENTATION_RESULTS_FOLDER)
if not os.path.isdir(seg_results_fldr):
    os.mkdir(seg_results_fldr)
seg_eval_fldr = os.path.join(extracted_folder, manager_stage1_results.SEGMENTATION_EVAL_FOLDER)
if not os.path.isdir(seg_eval_fldr):
    os.mkdir(seg_eval_fldr)
img_transformed_fldr = os.path.join(extracted_folder, "transformed_segmentations")
if not os.path.isdir(img_transformed_fldr):
    os.mkdir(img_transformed_fldr)

def copy_mapping(dataset_name:str, series_id: int, organs_info: pd.DataFrame, dest_organs_folder: str):
    patient_id = series_meta.loc[series_id, "patient_id"]
    series_folder = os.path.join("data_hdf5_cropped", str(patient_id), str(series_id))

    # copy the images
    z_poses = np.load(os.path.join(series_folder, "z_positions.npy"))
    ct_3D_image = h5py.File(os.path.join(series_folder, "ct_3D_image.hdf5"), "r")

    is_flipped = np.mean(np.diff(z_poses)) > 0.0

    for organ_id in range(3):
        if organs_info.loc[organs[organ_id], "found"]:
            min_slice, max_slice = organs_info.loc[organs[organ_id], "left"], organs_info.loc[organs[organ_id], "right"]
            z_min, z_max = z_poses[min_slice], z_poses[max_slice]
            expected_zposes = np.linspace(z_min, z_max, SAMPLING_DEPTH)
            if is_flipped:
                nearest_slice_indices = image_ROI_sampler.find_closest(z_poses, expected_zposes)
            else:
                nearest_slice_indices = (z_poses.shape[0] - 1 - image_ROI_sampler.find_closest(
                    z_poses[::-1], expected_zposes))[::-1]

            nearest_slice_indices = np.clip(nearest_slice_indices, 0, len(z_poses) - 1)

            # Flip the nearest slice indices if not flipped
            if not is_flipped:
                nearest_slice_indices = nearest_slice_indices[::-1]
            assert np.all(np.diff(nearest_slice_indices) >=0), "Nearest slice indices are not sorted"

            # Create the folder for transformed segmentations
            transformed_img_folder = os.path.join(img_transformed_fldr, dataset_name + "_" + organs[organ_id],
                                                            "data_hdf5_cropped", str(patient_id), str(series_id))
            if not os.path.isdir(transformed_img_folder):
                os.makedirs(transformed_img_folder)

            # Copy the images
            collapsed_nearest_indices, repeats = image_ROI_sampler.consecutive_repeats(nearest_slice_indices)
            cropped_3d = ct_3D_image["ct_3D_image"][collapsed_nearest_indices, ...]
            cropped_z_poses = z_poses[collapsed_nearest_indices]

            with h5py.File(os.path.join(transformed_img_folder, "ct_3D_image.hdf5"), "w") as f:
                f.create_dataset("ct_3D_image", data=cropped_3d, compression="gzip", compression_opts=0)
            np.save(os.path.join(transformed_img_folder, "z_positions.npy"), cropped_z_poses)

            # Set new limits
            organs_info.loc[organs[organ_id], "left"] = 0
            organs_info.loc[organs[organ_id], "right"] = len(collapsed_nearest_indices) - 1

    ct_3D_image.close()
    # save the organs information
    organs_info.to_csv(os.path.join(dest_organs_folder, str(series_id) + ".csv"))


if __name__ == "__main__":
    # Extracts the organ information from the stage 1 results
    parser = argparse.ArgumentParser(description='Extracts the images and labels from the stage 1 results to be placed into a folder for training with fastai')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to extract', required=True)

    args = parser.parse_args()
    datasets = args.datasets

    # loop through the datasets and extract the results
    for dataset in datasets:
        mgr = manager_stage1_results.Stage1ResultsManager(dataset)
        mgr.create_copy(seg_results_fldr, seg_eval_fldr, copy_mapping)


