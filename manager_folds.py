import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

folds_dataset_folder = "folds"
if not os.path.exists(folds_dataset_folder):
    os.makedirs(folds_dataset_folder)

patient_injuries = pd.read_csv("data/train.csv", index_col=0)

def dataset_exists(dataset_name: str):
    return os.path.isfile(os.path.join(folds_dataset_folder, dataset_name + ".json"))

def load_dataset(dataset_name: str) -> list[int]:
    with open(os.path.join(folds_dataset_folder, dataset_name + ".json"), "r") as f:
        cfg = json.load(f)

    return [int(patient_id) for patient_id in cfg["dataset"]]

def save_dataset(dataset_name: str, dataset: list[int]):
    with open(os.path.join(folds_dataset_folder, dataset_name + ".json"), "w") as f:
        json.dump({"dataset": dataset}, f, indent=4)

def add_argparse_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.parse_args()

def parse_args(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    train_data = args.train_data
    val_data = args.val_data
    assert dataset_exists(train_data)
    assert dataset_exists(val_data)

    train_data = load_dataset(train_data)
    val_data = load_dataset(val_data)

    # assert that they are disjoint
    assert len(set(train_data).intersection(set(val_data))) == 0

    return train_data, val_data

def get_summarization(body_part: str):
    assert body_part in ["bowel", "extravasation", "kidney", "liver", "spleen"]
    if body_part == "bowel":
        return patient_injuries["bowel_injury"]
    if body_part == "extravasation":
        return patient_injuries["extravasation_injury"]
    if body_part == "kidney":
        return patient_injuries["kidney_low"] + 2 * patient_injuries["kidney_high"]
    if body_part == "liver":
        return patient_injuries["liver_low"] + 2 * patient_injuries["liver_high"]
    if body_part == "spleen":
        return patient_injuries["spleen_low"] + 2 * patient_injuries["spleen_high"]

def get_summarization_string(body_parts: dict[str, bool]):
    label_str = ""
    summary = pd.Series("", index = patient_injuries.index)
    for body_part in body_parts:
        if body_parts[body_part]:
            if label_str != "":
                label_str += "_"
                summary = summary + "_"
            label_str += body_part
            summary = summary + get_summarization(body_part).astype(str)

    return label_str, summary

def get_bowel_status(patient_ids: list[int]) -> pd.Series:
    return patient_injuries["bowel_injury"].loc[patient_ids]

def get_extravasation_status(patient_ids: list[int]) -> pd.Series:
    return patient_injuries["extravasation_injury"].loc[patient_ids]

def get_kidney_status(patient_ids: list[int]) -> pd.DataFrame:
    return patient_injuries[["kidney_healthy", "kidney_low", "kidney_high"]].loc[patient_ids]

def get_liver_status(patient_ids: list[int]) -> pd.DataFrame:
    return patient_injuries[["liver_healthy", "liver_low", "liver_high"]].loc[patient_ids]

def get_spleen_status(patient_ids: list[int]) -> pd.DataFrame:
    return patient_injuries[["spleen_healthy", "spleen_low", "spleen_high"]].loc[patient_ids]

def get_kidney_status_binary(patient_ids: list[int]) -> pd.Series:
    return patient_injuries["kidney_low"].loc[patient_ids] + patient_injuries["kidney_high"].loc[patient_ids]

def get_liver_status_binary(patient_ids: list[int]) -> pd.Series:
    return patient_injuries["liver_low"].loc[patient_ids] + patient_injuries["liver_high"].loc[patient_ids]

def get_spleen_status_binary(patient_ids: list[int]) -> pd.Series:
    return patient_injuries["spleen_low"].loc[patient_ids] + patient_injuries["spleen_high"].loc[patient_ids]

def get_patient_status(patient_ids: list[int], labels: dict[str, object]) -> dict[str, np.ndarray]:
    """
    Returns a numpy array for each required label. If the label is binary, returns a (n,) array of 0s and 1s.
    If the label is multiclass, returns a (n, 3) array in one-hot encoding.
    """
    columns = {}
    if labels["bowel"]:
        columns["bowel"] = get_bowel_status(patient_ids).values
    if labels["extravasation"]:
        columns["extravasation"] = get_extravasation_status(patient_ids).values
    if labels["kidney"] == 2:
        columns["kidney"] = get_kidney_status(patient_ids).values
    elif labels["kidney"] == 1:
        columns["kidney"] = get_kidney_status_binary(patient_ids).values
    if labels["liver"] == 2:
        columns["liver"] = get_liver_status(patient_ids).values
    elif labels["liver"] == 1:
        columns["liver"] = get_liver_status_binary(patient_ids).values
    if labels["spleen"] == 2:
        columns["spleen"] = get_spleen_status(patient_ids).values
    elif labels["spleen"] == 1:
        columns["spleen"] = get_spleen_status_binary(patient_ids).values
    return columns

def get_patient_status_labels(patient_ids: list[int], labels: dict[str, object]) -> dict[str, np.ndarray]:
    """
    Returns a numpy array for each required label. Returns a (n,) array for each label regardless of the type of label.
    The array contains values 0, 1, 2 if multiclass, and values 0, 1 if binary.
    """
    columns = get_patient_status(patient_ids, labels)
    for column in columns:
        if len(columns[column].shape) > 1:
            columns[column] = np.argmax(columns[column], axis=1)
    return columns

def randomly_pick_series(patient_ids: list) -> list:
    series_ids = []
    for patient_id in patient_ids:
        patient_folder = os.path.join("data_hdf5", str(patient_id))
        series = os.listdir(patient_folder)
        series_ids.append(series[np.random.randint(len(series))])
    return series_ids

if __name__ == "__main__":
    label_str, summary = get_summarization_string({"bowel": False, "extravasation": False, "kidney": True, "liver": False, "spleen": True})
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    for k, (train_idx, val_idx) in enumerate(kfold.split(summary, summary)):
        train_data = summary.iloc[train_idx].index
        val_data = summary.iloc[val_idx].index
        save_dataset("kidney_spleen_train_fold_{}".format(k), [int(patient_id) for patient_id in train_data])
        save_dataset("kidney_spleen_val_fold_{}".format(k), [int(patient_id) for patient_id in val_data])
