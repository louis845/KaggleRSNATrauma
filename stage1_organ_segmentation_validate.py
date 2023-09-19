import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import tqdm
import h5py

import manager_stage1_results
import manager_segmentations
import manager_folds
import config
import metrics

def predict(predictions_folder: str, output_folder):
    """
    Predict on all patients in the training set
    """
    predictions = [int(filename[:-4]) for filename in os.listdir(predictions_folder) if filename.endswith(".csv")]

    slicewise_metrics = metrics.BinaryMetrics("slicewise")
    imagewise_metrics = metrics.BinaryMetrics("imagewise")

    organ_slicewise_metrics = []
    organ_imagewise_metrics = []
    organs = ["liver", "spleen", "kidney", "bowel"]
    for organ in organs:
        organ_slicewise_metrics.append(metrics.BinaryMetrics("slicewise_" + organ))
        organ_imagewise_metrics.append(metrics.BinaryMetrics("imagewise_" + organ))

    entrywise_preds = {
        "slice_iou": [],
        "slice_accuracy": [],
        "slice_precision": [],
        "slice_recall": [],
        "image_accuracy": [],
        "image_precision": [],
        "image_recall": [],
    }
    for organ in organs:
        entrywise_preds["slice_iou_" + organ] = []
        entrywise_preds["slice_accuracy_" + organ] = []
        entrywise_preds["slice_precision_" + organ] = []
        entrywise_preds["slice_recall_" + organ] = []
        entrywise_preds["image_accuracy_" + organ] = []
        entrywise_preds["image_precision_" + organ] = []
        entrywise_preds["image_recall_" + organ] = []

    for series_id in tqdm.tqdm(predictions):
        gt_segmentation_file = os.path.join(manager_segmentations.EXPERT_SEGMENTATION_FOLDER, str(series_id) + ".hdf5")
        if not os.path.isfile(gt_segmentation_file):
            gt_segmentation_file = os.path.join(manager_segmentations.TSM_SEGMENTATION_FOLDER, str(series_id) + ".hdf5")
        with h5py.File(gt_segmentation_file, "r") as f:
            gt_segmentation = torch.tensor(np.array(f["segmentation_arr"]), dtype=torch.bool, device=config.device)
        gt_segmentation[..., 2] = torch.logical_or(gt_segmentation[..., 2], gt_segmentation[..., 3])
        gt_segmentation[..., 3] = gt_segmentation[..., 4]
        gt_segmentation = gt_segmentation[..., :4].permute(0, 3, 1, 2)

        gt_image = torch.any(gt_segmentation, dim=0).to(torch.long)
        gt_slice = torch.any(torch.any(gt_segmentation, dim=-1), dim=-1).to(torch.long)

        slice_preds = pd.read_csv(os.path.join(predictions_folder, str(series_id) + ".csv"), index_col=0)
        slice_preds_array = np.zeros((gt_slice.shape[0], 4), dtype=bool)
        for i in range(4):
            if slice_preds.iloc[i]["found"]:
                slice_preds_array[slice_preds.iloc[i]["left"]:slice_preds.iloc[i]["right"], i] = True
        slice_preds_array = torch.tensor(slice_preds_array, dtype=torch.long, device=config.device)

        with h5py.File(os.path.join(predictions_folder, str(series_id) + ".hdf5"), "r") as f:
            image_preds = f["organ_location"][()]
        image_preds = torch.tensor(image_preds, dtype=torch.long, device=config.device)

        slicewise_metrics.add(slice_preds_array, gt_slice)
        imagewise_metrics.add(image_preds, gt_image)

        for i, organ in enumerate(organs):
            organ_slicewise_metrics[i].add(slice_preds_array[:, i], gt_slice[:, i])
            organ_imagewise_metrics[i].add(image_preds[i, ...], gt_image[i, ...])

        temp_slicewise_metrics = metrics.BinaryMetrics("temp_slicewise")
        temp_imagewise_metrics = metrics.BinaryMetrics("temp_imagewise")
        temp_slicewise_metrics.add(slice_preds_array, gt_slice)
        temp_imagewise_metrics.add(image_preds, gt_image)

        iou_slicewise = temp_slicewise_metrics.get_iou()
        slice_accuracy, slice_precision, slice_recall = temp_slicewise_metrics.get()
        accuracy, precision, recall = temp_imagewise_metrics.get()
        entrywise_preds["slice_iou"].append(iou_slicewise)
        entrywise_preds["slice_accuracy"].append(slice_accuracy)
        entrywise_preds["slice_precision"].append(slice_precision)
        entrywise_preds["slice_recall"].append(slice_recall)
        entrywise_preds["image_accuracy"].append(accuracy)
        entrywise_preds["image_precision"].append(precision)
        entrywise_preds["image_recall"].append(recall)

        for i, organ in enumerate(organs):
            temp_slicewise_metrics = metrics.BinaryMetrics("temp_slicewise")
            temp_imagewise_metrics = metrics.BinaryMetrics("temp_imagewise")
            temp_slicewise_metrics.add(slice_preds_array[:, i], gt_slice[:, i])
            temp_imagewise_metrics.add(image_preds[i, ...], gt_image[i, ...])

            iou_slicewise = temp_slicewise_metrics.get_iou()
            slice_accuracy, slice_precision, slice_recall = temp_slicewise_metrics.get()
            accuracy, precision, recall = temp_imagewise_metrics.get()

            entrywise_preds["slice_iou_" + organ].append(iou_slicewise)
            entrywise_preds["slice_accuracy_" + organ].append(slice_accuracy)
            entrywise_preds["slice_precision_" + organ].append(slice_precision)
            entrywise_preds["slice_recall_" + organ].append(slice_recall)
            entrywise_preds["image_accuracy_" + organ].append(accuracy)
            entrywise_preds["image_precision_" + organ].append(precision)
            entrywise_preds["image_recall_" + organ].append(recall)

    # save entrywise predictions
    pd.DataFrame(entrywise_preds, index=predictions).to_csv(os.path.join(output_folder, "entrywise_preds.csv"))

    # save metrics
    with open(os.path.join(output_folder, "metrics.txt"), "w") as f:
        slicewise_metrics.report_print_to_file(f, True)
        imagewise_metrics.report_print_to_file(f, True)
        for i, organ in enumerate(organs):
            organ_slicewise_metrics[i].report_print_to_file(f, True)
            organ_imagewise_metrics[i].report_print_to_file(f, True)


def validate_folder_preds_full(data: list, predictions_folder: str):
    for patient_id in data:
        patient_folder = os.path.join("data_hdf5_cropped", str(patient_id))
        for series_id in os.listdir(patient_folder):
            assert os.path.isfile(os.path.join(predictions_folder, series_id + ".hdf5")), "Missing prediction for series " + series_id
            assert os.path.isfile(os.path.join(predictions_folder, series_id + ".csv")), "Missing prediction for series " + series_id

            assert os.path.isfile(os.path.join(manager_segmentations.EXPERT_SEGMENTATION_FOLDER, series_id + ".hdf5")) or\
                os.path.isfile(os.path.join(manager_segmentations.TSM_SEGMENTATION_FOLDER, series_id + ".hdf5")), "Missing segmentations for series " + series_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    manager_folds.add_argparse_arguments(parser)
    config.add_argparse_arguments(parser)

    args = parser.parse_args()
    config.parse_args(args)
    train_data, val_data, train_data_name, val_data_name = manager_folds.parse_args(args)

    folder = manager_stage1_results.SEGMENTATION_RESULTS_FOLDER
    out_folder = manager_stage1_results.SEGMENTATION_EVAL_FOLDER

    # eval on training set
    print("Eval on training set")
    train_preds_folder = os.path.join(folder, train_data_name)
    train_out_folder = os.path.join(out_folder, train_data_name)
    if os.path.isdir(train_out_folder):
        shutil.rmtree(train_out_folder)
    os.makedirs(train_out_folder)
    validate_folder_preds_full(train_data, train_preds_folder)
    predict(train_preds_folder, train_out_folder)

    # eval on validation set
    print("Eval on validation set")
    val_preds_folder = os.path.join(folder, val_data_name)
    val_out_folder = os.path.join(out_folder, val_data_name)
    if os.path.isdir(val_out_folder):
        shutil.rmtree(val_out_folder)
    os.makedirs(val_out_folder)
    validate_folder_preds_full(val_data, val_preds_folder)
    predict(val_preds_folder, val_out_folder)
