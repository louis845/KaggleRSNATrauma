import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import manager_segmentations

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

def parse_args(args: argparse.Namespace) -> tuple[list[int], list[int], str, str]:
    train_data_name = args.train_data
    val_data_name = args.val_data
    assert dataset_exists(train_data_name)
    assert dataset_exists(val_data_name)

    train_data = load_dataset(train_data_name)
    val_data = load_dataset(val_data_name)

    # assert that they are disjoint
    assert len(set(train_data).intersection(set(val_data))) == 0

    return train_data, val_data, train_data_name, val_data_name

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

def get_bowel_status_single(patient_id: int) -> int:
    return int(patient_injuries["bowel_injury"].loc[patient_id])

def get_extravasation_status_single(patient_id: int) -> int:
    return int(patient_injuries["extravasation_injury"].loc[patient_id])

def get_kidney_status_single(patient_id: int) -> np.ndarray:
    return patient_injuries[["kidney_healthy", "kidney_low", "kidney_high"]].loc[patient_id].values

def get_liver_status_single(patient_id: int) -> np.ndarray:
    return patient_injuries[["liver_healthy", "liver_low", "liver_high"]].loc[patient_id].values

def get_spleen_status_single(patient_id: int) -> np.ndarray:
    return patient_injuries[["spleen_healthy", "spleen_low", "spleen_high"]].loc[patient_id].values

def get_patient_status_single(patient_id: int, class_code: int, is_ternary: bool) -> int:
    """
    Get the status of a patient for a specific class.
    :param patient_id: The patient id
    :param class_code: The class code. 0 - liver, 1 - spleen, 2 - kidney, 3 - bowel, 4 - extravasation
    :param is_ternary: Whether to return a ternary or binary label. Ignored if bowel or extravasation, as they are binary
    :return: The status of the patient for the class
    """
    if class_code == 0:
        if is_ternary:
            return int(patient_injuries.loc[patient_id, "liver_low"] + 2 * patient_injuries.loc[patient_id, "liver_high"])
        else:
            return 1 - int(patient_injuries.loc[patient_id, "liver_healthy"])
    if class_code == 1:
        if is_ternary:
            return int(patient_injuries.loc[patient_id, "spleen_low"] + 2 * patient_injuries.loc[patient_id, "spleen_high"])
        else:
            return 1 - int(patient_injuries.loc[patient_id, "spleen_healthy"])
    if class_code == 2:
        if is_ternary:
            return int(patient_injuries.loc[patient_id, "kidney_low"] + 2 * patient_injuries.loc[patient_id, "kidney_high"])
        else:
            return 1 - int(patient_injuries.loc[patient_id, "kidney_healthy"])
    if class_code == 3:
        return int(patient_injuries.loc[patient_id, "bowel_injury"])
    if class_code == 4:
        return int(patient_injuries.loc[patient_id, "extravasation_injury"])
    raise ValueError("Invalid class code")

def has_injury(patient_id: int) -> bool:
    return int(patient_injuries.loc[patient_id, "any_injury"]) == 1

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

def randomly_pick_series(patient_ids: list, data_folder="data_hdf5") -> list:
    series_ids = []
    for patient_id in patient_ids:
        patient_folder = os.path.join(data_folder, str(patient_id))
        series = os.listdir(patient_folder)
        series_ids.append(series[np.random.randint(len(series))])
    return series_ids

if __name__ == "__main__":
    if not os.path.isfile("folds/small_kidney_fold_0.json"):
        label_str, summary = get_summarization_string({"bowel": False, "extravasation": False, "kidney": True, "liver": False, "spleen": False})
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        for k, (train_idx, val_idx) in enumerate(kfold.split(summary, summary)):
            folds_data = summary.iloc[val_idx].index
            save_dataset("small_kidney_fold_{}".format(k), [int(patient_id) for patient_id in folds_data])
    if not os.path.isfile("folds/kidney_spleen_train_fold_0.json"):
        label_str, summary = get_summarization_string({"bowel": False, "extravasation": False, "kidney": True, "liver": False, "spleen": True})
        kfold = StratifiedKFold(n_splits=3, shuffle=True)
        for k, (train_idx, val_idx) in enumerate(kfold.split(summary, summary)):
            train_data = summary.iloc[train_idx].index
            val_data = summary.iloc[val_idx].index
            save_dataset("kidney_spleen_train_fold_{}".format(k), [int(patient_id) for patient_id in train_data])
            save_dataset("kidney_spleen_val_fold_{}".format(k), [int(patient_id) for patient_id in val_data])
    if not os.path.isfile("folds/small_kidney_spleen_fold_0.json"):
        label_str, summary = get_summarization_string({"bowel": False, "extravasation": False, "kidney": True, "liver": False, "spleen": True})
        kfold = StratifiedKFold(n_splits=20, shuffle=True)
        for k, (train_idx, val_idx) in enumerate(kfold.split(summary, summary)):
            folds_data = summary.iloc[val_idx].index
            save_dataset("small_kidney_spleen_fold_{}".format(k), [int(patient_id) for patient_id in folds_data])

    if not os.path.isfile("folds/tiny_kidney_train.json"):
        train_ids = []
        val_ids = []
        num_train_kidney1 = 0
        num_train_kidney0 = 0
        num_val_kidney1 = 0
        num_val_kidney0 = 0
        idx = 0
        while (num_val_kidney0 < 10) or (num_val_kidney1 < 10):
            patient_id = patient_injuries.index[idx]
            if patient_injuries["kidney_healthy"][patient_id] == 1:
                if num_train_kidney1 < 10:
                    train_ids.append(int(patient_id))
                    num_train_kidney1 += 1
                elif num_val_kidney1 < 10:
                    val_ids.append(int(patient_id))
                    num_val_kidney1 += 1
            else:
                if num_train_kidney0 < 10:
                    train_ids.append(int(patient_id))
                    num_train_kidney0 += 1
                elif num_val_kidney0 < 10:
                    val_ids.append(int(patient_id))
                    num_val_kidney0 += 1
            idx += 1

        save_dataset("tiny_kidney_train", train_ids)
        save_dataset("tiny_kidney_val", val_ids)

    if not os.path.isfile("split10.json"):
        patient_ids_all = list(patient_injuries.index)
        patient_ids_split10 = {}
        for k in range(10):
            patient_ids_split10["split{}".format(k)] = patient_ids_all[k::10]

        with open("split10.json", "w") as f:
            json.dump(patient_ids_split10, f, indent=4)

    if not os.path.isfile("folds/segmentation_fold1_train.json"):
        label_str, summary = get_summarization_string(
            {"bowel": True, "extravasation": True, "kidney": True, "liver": True, "spleen": True})

        # get and check segmentation data
        patients_with_segmentation = [int(x) for x in manager_segmentations.get_patients_with_expert_segmentation()]

        # restrict to those with segmentations and split
        summary = summary.loc[patients_with_segmentation]
        kfold = StratifiedKFold(n_splits=3, shuffle=True)
        for k, (train_idx, val_idx) in enumerate(kfold.split(summary, summary)):
            train_data = summary.iloc[train_idx].index
            val_data = summary.iloc[val_idx].index
            save_dataset("segmentation_fold{}_train".format(k), [int(patient_id) for patient_id in train_data])
            save_dataset("segmentation_fold{}_val".format(k), [int(patient_id) for patient_id in val_data])

    if not os.path.isfile("folds/segmentation_extra_fold1_train.json"):
        label_str, summary = get_summarization_string(
            {"bowel": True, "extravasation": True, "kidney": True, "liver": True, "spleen": True})

        # get and check segmentation data
        patients_with_expert_segmentation = [int(x) for x in manager_segmentations.get_patients_with_expert_segmentation()]
        patients_with_TSM_segmentation = [int(x) for x in manager_segmentations.get_patients_with_TSM_segmentation() if int(x) != 15472]

        # restrict to those with segmentations and split
        summary = summary.loc[patients_with_expert_segmentation]
        kfold = StratifiedKFold(n_splits=3, shuffle=True)
        for k, (train_idx, val_idx) in enumerate(kfold.split(summary, summary)):
            train_data = summary.iloc[train_idx].index
            val_data = summary.iloc[val_idx].index
            save_dataset("segmentation_extra_fold{}_train".format(k), [int(patient_id) for patient_id in train_data] + patients_with_TSM_segmentation)
            save_dataset("segmentation_extra_fold{}_val".format(k), [int(patient_id) for patient_id in val_data])

    if not os.path.isfile("folds/ROI_classifier_fold0_train.json"):
        label_str, summary = get_summarization_string(
            {"bowel": True, "extravasation": True, "kidney": True, "liver": True, "spleen": True})
        # stratify by whether patient has expert segmentation also
        with_expert_segmentation = manager_segmentations.get_patients_with_expert_segmentation()
        has_segmentation_summary = []
        for patient_id in summary.index:
            if str(patient_id) in with_expert_segmentation:
                has_segmentation_summary.append("_HasSegmentation")
            else:
                has_segmentation_summary.append("_NoSegmentation")
        has_segmentation_summary = pd.Series(has_segmentation_summary, index=summary.index)
        summary = summary + has_segmentation_summary

        kfold = StratifiedKFold(n_splits=3, shuffle=True)
        for k, (train_idx, val_idx) in enumerate(kfold.split(summary, summary)):
            train_data = summary.iloc[train_idx].index
            val_data = summary.iloc[val_idx].index
            save_dataset("ROI_classifier_fold{}_train".format(k), [int(patient_id) for patient_id in train_data])
            save_dataset("ROI_classifier_fold{}_val".format(k), [int(patient_id) for patient_id in val_data])

    if not os.path.isfile("folds/ROI_classifier_fold0_expert_train.json"):
        with_expert_segmentation = manager_segmentations.get_patients_with_expert_segmentation()
        for k in range(3):
            train_entries = load_dataset("ROI_classifier_fold{}_train".format(k))
            val_entries = load_dataset("ROI_classifier_fold{}_val".format(k))
            train_entries = [x for x in train_entries if str(x) in with_expert_segmentation]
            val_entries = [x for x in val_entries if str(x) in with_expert_segmentation]
            save_dataset("ROI_classifier_fold{}_expert_train".format(k), train_entries)
            save_dataset("ROI_classifier_fold{}_expert_val".format(k), val_entries)
