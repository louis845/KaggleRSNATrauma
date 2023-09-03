import pandas as pd
import os
import numpy as np

shape_info = pd.read_csv("data_hdf5_cropped/shape_info.csv", index_col=1)
series_meta = pd.read_csv("data/train_series_meta.csv", index_col=1)

EXPERT_SEGMENTATION_FOLDER = "data_segmentation_hdf_cropped"
TSM_SEGMENTATION_FOLDER = "total_segmentator_hdf_cropped"

def get_mean_slope(series_id):
    return shape_info.loc[int(series_id), "mean_slope"]

def patient_has_expert_segmentations(patient_id):
    relevant_segmentations = list(series_meta.loc[series_meta["patient_id"] == int(patient_id)].index)
    for series_id in relevant_segmentations:
        if os.path.isfile(os.path.join(EXPERT_SEGMENTATION_FOLDER, str(series_id) + ".hdf5")):
            return True
    return False

def patient_has_TSM_segmentations(patient_id):
    relevant_segmentations = list(series_meta.loc[series_meta["patient_id"] == int(patient_id)].index)
    for series_id in relevant_segmentations:
        if os.path.isfile(os.path.join(TSM_SEGMENTATION_FOLDER, str(series_id) + ".hdf5")):
            return True
    return False

def randomly_pick_expert_segmentation(patient_id):
    all_segmentations = list(series_meta.loc[series_meta["patient_id"] == int(patient_id)].index)
    relevant_segmentations = []
    for series_id in all_segmentations:
        if os.path.isfile(os.path.join(EXPERT_SEGMENTATION_FOLDER, str(series_id) + ".hdf5")):
            relevant_segmentations.append(series_id)
    assert len(relevant_segmentations) > 0, "No expert segmentations found for patient " + str(patient_id)
    return relevant_segmentations[np.random.randint(len(relevant_segmentations))]

def randomly_pick_TSM_segmentation(patient_id):
    all_segmentations = list(series_meta.loc[series_meta["patient_id"] == int(patient_id)].index)
    relevant_segmentations = []
    for series_id in all_segmentations:
        if os.path.isfile(os.path.join(TSM_SEGMENTATION_FOLDER, str(series_id) + ".hdf5")):
            relevant_segmentations.append(series_id)
    assert len(relevant_segmentations) > 0, "No TSM segmentations found for patient " + str(patient_id)
    return relevant_segmentations[np.random.randint(len(relevant_segmentations))]

def get_patients_with_expert_segmentation() -> list[str]:
    expert_segmentation_series_id = [int(x[:-5]) for x in os.listdir(EXPERT_SEGMENTATION_FOLDER)]
    patient_ids = list(series_meta.loc[expert_segmentation_series_id, "patient_id"].unique())
    return [str(x) for x in patient_ids]

def get_patients_with_TSM_segmentation(only_negative_slope=False) -> list[str]:
    TSM_segmentation_series_id = [int(x[:-5]) for x in os.listdir(TSM_SEGMENTATION_FOLDER)]
    if only_negative_slope:
        TSM_segmentation_series_id = [x for x in TSM_segmentation_series_id if get_mean_slope(x) < 0]
    patient_ids = list(series_meta.loc[TSM_segmentation_series_id, "patient_id"].unique())
    return [str(x) for x in patient_ids]
