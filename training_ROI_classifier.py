import gc
import os
import time
import argparse
import json
import traceback
import multiprocessing
import collections

import pandas as pd
import numpy as np
import tqdm

import torch
import torch.nn

import config
import logging_memory_utils
import manager_folds
import manager_models
import model_3d_patch_resnet
import metrics
import image_ROI_sampler
import image_ROI_sampler_async
import manager_segmentations
import training_shuffle_utils


class MetricKeys:
    METRIC_TYPE_LOSS = "loss"
    METRIC_TYPE_INJURY = "injury"
    METRIC_TYPE_SLICE_INJURY = "slice_injury"

    LOSS = "loss"
    LIVER = "liver"
    LIVER_LOSS = "liver_loss"
    SPLEEN = "spleen"
    SPLEEN_LOSS = "spleen_loss"
    KIDNEY = "kidney"
    KIDNEY_LOSS = "kidney_loss"
    BOWEL = "bowel"
    BOWEL_LOSS = "bowel_loss"

    INJURY_LOSS = "injury_loss"
    LIVER_INJURY = "liver_injury"
    LIVER_SLICE_INJURY = "liver_slice_injury"
    LIVER_INJURY_LOSS = "liver_injury_loss"
    SPLEEN_INJURY = "spleen_injury"
    SPLEEN_SLICE_INJURY = "spleen_slice_injury"
    SPLEEN_INJURY_LOSS = "spleen_injury_loss"
    KIDNEY_INJURY = "kidney_injury"
    KIDNEY_SLICE_INJURY = "kidney_slice_injury"
    KIDNEY_INJURY_LOSS = "kidney_injury_loss"
    BOWEL_INJURY = "bowel_injury"
    BOWEL_SLICE_INJURY = "bowel_slice_injury"
    BOWEL_INJURY_LOSS = "bowel_injury_loss"

    @staticmethod
    def get_segmentation_metric_key_by_class_code(class_code: int, is_loss: bool):
        if class_code == 0:
            if is_loss:
                return MetricKeys.LIVER_LOSS
            else:
                return MetricKeys.LIVER
        elif class_code == 1:
            if is_loss:
                return MetricKeys.SPLEEN_LOSS
            else:
                return MetricKeys.SPLEEN
        elif class_code == 2:
            if is_loss:
                return MetricKeys.KIDNEY_LOSS
            else:
                return MetricKeys.KIDNEY
        elif class_code == 3:
            if is_loss:
                return MetricKeys.BOWEL_LOSS
            else:
                return MetricKeys.BOWEL
        else:
            raise ValueError("Invalid class code")

    @staticmethod
    def get_injury_metric_key_by_class_code(class_code: int, metric_type: str):
        assert metric_type in [MetricKeys.METRIC_TYPE_LOSS, MetricKeys.METRIC_TYPE_INJURY, MetricKeys.METRIC_TYPE_SLICE_INJURY]
        if class_code == 0:
            if metric_type == MetricKeys.METRIC_TYPE_LOSS:
                return MetricKeys.LIVER_INJURY_LOSS
            elif metric_type == MetricKeys.METRIC_TYPE_INJURY:
                return MetricKeys.LIVER_INJURY
            elif metric_type == MetricKeys.METRIC_TYPE_SLICE_INJURY:
                return MetricKeys.LIVER_SLICE_INJURY
        elif class_code == 1:
            if metric_type == MetricKeys.METRIC_TYPE_LOSS:
                return MetricKeys.SPLEEN_INJURY_LOSS
            elif metric_type == MetricKeys.METRIC_TYPE_INJURY:
                return MetricKeys.SPLEEN_INJURY
            elif metric_type == MetricKeys.METRIC_TYPE_SLICE_INJURY:
                return MetricKeys.SPLEEN_SLICE_INJURY
        elif class_code == 2:
            if metric_type == MetricKeys.METRIC_TYPE_LOSS:
                return MetricKeys.KIDNEY_INJURY_LOSS
            elif metric_type == MetricKeys.METRIC_TYPE_INJURY:
                return MetricKeys.KIDNEY_INJURY
            elif metric_type == MetricKeys.METRIC_TYPE_SLICE_INJURY:
                return MetricKeys.KIDNEY_SLICE_INJURY
        elif class_code == 3:
            if metric_type == MetricKeys.METRIC_TYPE_LOSS:
                return MetricKeys.BOWEL_INJURY_LOSS
            elif metric_type == MetricKeys.METRIC_TYPE_INJURY:
                return MetricKeys.BOWEL_INJURY
            elif metric_type == MetricKeys.METRIC_TYPE_SLICE_INJURY:
                return MetricKeys.BOWEL_SLICE_INJURY

class UsedLabelManager:
    levels_used = [2, 2, 2]

    @staticmethod
    def get_liver_level():
        return UsedLabelManager.levels_used[0]

    @staticmethod
    def get_spleen_level():
        return UsedLabelManager.levels_used[1]

    @staticmethod
    def get_kidney_level():
        return UsedLabelManager.levels_used[2]

    @staticmethod
    def is_ternary(class_code: int):
        if class_code == 3: # bowel is always binary
            return False
        return UsedLabelManager.levels_used[class_code] == 2


class TrainingTypes:
    SEGMENTATIONS = 0
    INJURIES = 1
    INJURIES_WITH_GUIDANCE = 2

class ShuffleKeys:
    PATIENT_ID = "patient_id"
    TRAINING_TYPE = "training_type"

def create_shuffle_entry(patient_id: int, training_type: int):
    assert training_type in [TrainingTypes.SEGMENTATIONS, TrainingTypes.INJURIES, TrainingTypes.INJURIES_WITH_GUIDANCE]
    return {ShuffleKeys.PATIENT_ID: patient_id, ShuffleKeys.TRAINING_TYPE: training_type}


def initialize_training_entries(training_entries: list[int], validation_entries: list[int], extra_nonexpert_segmentation_training_ratio):
    expert_training_entries = manager_segmentations.restrict_patients_to_expert_segmentation(training_entries)
    extra_training_entries = manager_segmentations.restrict_patients_to_TSM_but_no_expert_segmentation(training_entries)

    print("Detected {} expert training entries, {} extra training entries".format(len(expert_training_entries), len(extra_training_entries)))
    if len(expert_training_entries) < 50:
        raise ValueError("Not enough expert training entries.")

    injury_training_list = []
    for entry in training_entries:
        if manager_segmentations.patient_has_expert_segmentations(entry):
            injury_training_list.append(create_shuffle_entry(entry, TrainingTypes.INJURIES_WITH_GUIDANCE))
        else:
            injury_training_list.append(create_shuffle_entry(entry, TrainingTypes.INJURIES))
    segmentation_training_list = [create_shuffle_entry(entry, TrainingTypes.SEGMENTATIONS)
                                  for entry in expert_training_entries]
    segmentation_extra_training_list = [create_shuffle_entry(entry, TrainingTypes.SEGMENTATIONS)
                                        for entry in extra_training_entries]
    if len(extra_training_entries) > 0 and extra_nonexpert_segmentation_training_ratio > 0:
        training_entries = training_shuffle_utils.MultipleBiasedShuffleInfo(injury_training_list, [segmentation_training_list, segmentation_extra_training_list],
                                                         extra_ratio=1.0, within_extra_ratios=[1.0, extra_nonexpert_segmentation_training_ratio])
    else:
        training_entries = training_shuffle_utils.MultipleBiasedShuffleInfo(injury_training_list, [segmentation_training_list],
                                                         extra_ratio=1.0, within_extra_ratios=[1.0])

    validation_expert_entries = manager_segmentations.restrict_patients_to_expert_segmentation(validation_entries)
    validation_remaining_entries = [entry for entry in validation_entries if entry not in validation_expert_entries]

    assert len(validation_expert_entries) + len(validation_remaining_entries) == len(validation_entries)
    assert len(validation_expert_entries) >= 10, "Not enough expert validation entries."

    return training_entries, (validation_expert_entries, validation_remaining_entries)



bowel_mask_tensor = None  # to be initialized with tensor_2d3d


# focal loss with exponent 2
def focal_loss_segmentation(output: torch.Tensor, target: torch.Tensor, reduce_channels=True, include_bowel=True):
    # logsumexp trick for numerical stability
    binary_ce = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction="none") * (
                1 + target * (positive_weight - 1))
    if not include_bowel:
        binary_ce = binary_ce * bowel_mask_tensor
    if reduce_channels:
        return torch.mean(((target - torch.sigmoid(output)) ** 2) * binary_ce)
    else:
        return torch.mean(((target - torch.sigmoid(output)) ** 2) * binary_ce, dim=tensor_2d3d)

def binary_focal_loss_slicepreds(preds: torch.Tensor, target: torch.Tensor):
    # logsumexp trick for numerical stability
    target_float = target.to(torch.float32)
    binary_ce = torch.nn.functional.binary_cross_entropy_with_logits(preds, target_float, reduction="none")
    return torch.mean(((target_float - torch.sigmoid(preds)) ** 2) * binary_ce)

def ternary_focal_loss_slicepreds(preds: torch.Tensor, target: torch.Tensor):
    # logsumexp trick for numerical stability
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=3).to(torch.float32)
    ce = torch.nn.functional.cross_entropy(preds, target, reduction="none")
    probas = torch.softmax(preds, dim=1)
    focus = torch.sum((probas - target_one_hot) ** 2, dim=1)
    assert focus.shape == ce.shape
    return torch.mean(focus * ce)


def single_training_step_segmentation(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                         slices_: torch.Tensor, segmentations_: torch.Tensor, is_expert: bool):
    optimizer_.zero_grad()
    pred_probas, per_slice_logits, pred_segmentations = model_(slices_)
    with torch.no_grad():
        preds = (pred_segmentations > 0).to(torch.float32)
    # we include the bowel only for expert segmentations, and penalize non-expert segmentations only half as much
    loss = focal_loss_segmentation(pred_segmentations, segmentations_, reduce_channels=True, include_bowel=is_expert)
    if not is_expert:
        loss = loss * 0.5
    (loss * 0.0625).backward()
    optimizer.step()

    with torch.no_grad():
        tp_pixels = (preds * segmentations_).to(torch.long)
        tn_pixels = ((1 - preds) * (1 - segmentations_)).to(torch.long)
        fp_pixels = (preds * (1 - segmentations_)).to(torch.long)
        fn_pixels = ((1 - preds) * segmentations_).to(torch.long)

        tp_per_class = torch.sum(tp_pixels, dim=tensor_2d3d)
        tn_per_class = torch.sum(tn_pixels, dim=tensor_2d3d)
        fp_per_class = torch.sum(fp_pixels, dim=tensor_2d3d)
        fn_per_class = torch.sum(fn_pixels, dim=tensor_2d3d)

        loss_per_class = focal_loss_segmentation(pred_segmentations, segmentations_, reduce_channels=False)

    return loss, tp_per_class, tn_per_class, fp_per_class, fn_per_class, loss_per_class

def single_training_step_injury(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            slices: torch.Tensor, injury_labels: list[torch.Tensor], deep_guidance: bool):
    optimizer_.zero_grad()
    pred_probas, per_slice_logits, pred_segmentations = model_(slices)
    loss = 0
    deep_class_losses = []
    class_losses = []
    for k in range(4):
        is_ternary = UsedLabelManager.is_ternary(k)
        if deep_guidance:
            if is_ternary:
                class_loss = ternary_focal_loss_slicepreds(per_slice_logits[k], injury_labels[k])
            else:
                class_loss = binary_focal_loss_slicepreds(per_slice_logits[k], injury_labels[k].unsqueeze(-1))
            loss += (class_loss * 0.25)
            deep_class_losses.append(class_loss.item())

        if is_ternary:
            class_loss = torch.nn.functional.nll_loss(torch.log(torch.clamp(pred_probas[k], min=1e-10)), torch.max(injury_labels[k], dim=0, keepdim=True)[0], reduction="mean")
        else:
            class_loss = torch.nn.functional.binary_cross_entropy(pred_probas[k], torch.max(injury_labels[k], dim=0, keepdim=True)[0].unsqueeze(0).to(torch.float32), reduction="mean")
        loss += class_loss
        class_losses.append(class_loss.item())

    loss.backward()
    optimizer_.step()

    pred_classes = []
    per_slice_pred_classes = []
    with torch.no_grad():
        for k in range(4):
            is_ternary = UsedLabelManager.is_ternary(k)
            if is_ternary:
                pred_classes.append(torch.argmax(pred_probas[k], dim=1))
                per_slice_pred_classes.append(torch.argmax(per_slice_logits[k], dim=1))
            else:
                pred_classes.append((pred_probas[k] > 0.5).squeeze(-1).to(torch.long))
                per_slice_pred_classes.append((per_slice_logits[k] > 0).squeeze(-1).to(torch.long))

    return loss, deep_class_losses, class_losses, pred_classes, per_slice_pred_classes

def training_step(record: bool):
    if record:
        for key in train_metrics:
            train_metrics[key].reset()

    # shuffle
    shuffle_indices = training_entries.get_random_shuffle_indices()

    # training
    trained = 0
    with tqdm.tqdm(total=len(shuffle_indices)) as pbar:
        while trained < len(shuffle_indices):
            shuffle_info = training_entries[shuffle_indices[trained]]
            patient_id = shuffle_info[ShuffleKeys.PATIENT_ID]
            training_type = shuffle_info[ShuffleKeys.TRAINING_TYPE]

            # prepare options for image sampler
            if training_type == TrainingTypes.SEGMENTATIONS:
                if manager_segmentations.patient_has_expert_segmentations(patient_id):
                    series_id = str(manager_segmentations.randomly_pick_expert_segmentation(patient_id))
                    is_expert = True
                    seg_folder = manager_segmentations.EXPERT_SEGMENTATION_FOLDER
                    injury_labels_depth = -1
                else:
                    series_id = str(manager_segmentations.randomly_pick_TSM_segmentation(patient_id))
                    is_expert = False
                    seg_folder = manager_segmentations.TSM_SEGMENTATION_FOLDER
                    injury_labels_depth = -1
            elif training_type == TrainingTypes.INJURIES or training_type == TrainingTypes.INJURIES_WITH_GUIDANCE:
                any_injury = manager_folds.has_injury(patient_id)
                series_id = str(manager_folds.randomly_pick_series([patient_id], data_folder="data_hdf5_cropped")[0])
                seg_folder = None
                if any_injury:
                    injury_labels_depth = 5 if use_3d_prediction else 1
                else:
                    injury_labels_depth = -1

            # sample now
            if use_async_sampler:
                slices, segmentations, injury_labels = image_ROI_sampler_async.load_image(patient_id, series_id,
                                                                           segmentation_folder=seg_folder,
                                                                           slices_random=not disable_random_slices,
                                                                           translate_rotate_augmentation=not disable_rotpos_augmentation,
                                                                           elastic_augmentation=not disable_elastic_augmentation,
                                                                           slices=num_slices,
                                                                           segmentation_region_depth=5 if use_3d_prediction else 1,
                                                                           injury_labels_depth=injury_labels_depth)
            else:
                slices, segmentations, injury_labels = image_ROI_sampler.load_image(patient_id, series_id,
                                                                     segmentation_folder=seg_folder,
                                                                     slices_random=not disable_random_slices,
                                                                     translate_rotate_augmentation=not disable_rotpos_augmentation,
                                                                     elastic_augmentation=not disable_elastic_augmentation,
                                                                     slices=num_slices,
                                                                     segmentation_region_depth=5 if use_3d_prediction else 1,
                                                                     injury_labels_depth=injury_labels_depth)

            # do training now
            if training_type == TrainingTypes.SEGMENTATIONS:
                loss, tp_per_class, tn_per_class, fp_per_class, \
                    fn_per_class, loss_per_class = single_training_step_segmentation_compile(model, optimizer, slices, segmentations,
                                                                                is_expert=is_expert)
                loss = loss.item()
                tp_per_class = tp_per_class.cpu().numpy()
                tn_per_class = tn_per_class.cpu().numpy()
                fp_per_class = fp_per_class.cpu().numpy()
                fn_per_class = fn_per_class.cpu().numpy()
                loss_per_class = loss_per_class.cpu().numpy()

                # record
                if record:
                    # compute metrics
                    for class_code in range(4):
                        if class_code == 3 and not is_expert:
                            continue
                        organ_loss_key = MetricKeys.get_segmentation_metric_key_by_class_code(class_code, is_loss=True)
                        organ_key = MetricKeys.get_segmentation_metric_key_by_class_code(class_code, is_loss=False)
                        train_metrics[organ_loss_key].add(loss_per_class[class_code], 1)
                        train_metrics[organ_key].add_direct(tp_per_class[class_code], tn_per_class[class_code],
                                                            fp_per_class[class_code], fn_per_class[class_code])
                    train_metrics[MetricKeys.LOSS].add(loss, 1)
            elif (training_type == TrainingTypes.INJURIES or training_type == TrainingTypes.INJURIES_WITH_GUIDANCE):
                if not any_injury:
                    injury_labels = np.zeros((num_slices, 5), dtype=np.uint8)

                per_slice_class_labels = []
                for k in range(4):
                    if UsedLabelManager.is_ternary(k):
                        labels = torch.tensor(injury_labels[:, k], dtype=torch.long, device=config.device)
                    else:
                        labels = torch.tensor((injury_labels[:, k] > 0), dtype=torch.long, device=config.device)
                    per_slice_class_labels.append(labels)
                loss, deep_class_losses, class_losses, pred_classes, per_slice_pred_classes = single_training_step_injury_compile(model, optimizer, slices, per_slice_class_labels,
                                                                                                                          deep_guidance=(training_type == TrainingTypes.INJURIES_WITH_GUIDANCE))
                loss = loss.item()

                # record
                if record:
                    # compute metrics
                    for class_code in range(4):
                        organ_loss_key = MetricKeys.get_injury_metric_key_by_class_code(class_code, MetricKeys.METRIC_TYPE_LOSS)
                        organ_injury_key = MetricKeys.get_injury_metric_key_by_class_code(class_code, MetricKeys.METRIC_TYPE_INJURY)
                        organ_slice_injury_key = MetricKeys.get_injury_metric_key_by_class_code(class_code, MetricKeys.METRIC_TYPE_SLICE_INJURY)

                        train_metrics[organ_loss_key].add(class_losses[class_code], 1)
                        train_metrics[organ_injury_key].add(pred_classes[class_code], torch.max(per_slice_class_labels[class_code], dim=0, keepdim=True)[0])
                        train_metrics[organ_slice_injury_key].add(per_slice_pred_classes[class_code], per_slice_class_labels[class_code])
                    train_metrics[MetricKeys.INJURY_LOSS].add(loss, 1)

            trained += 1
            pbar.update(1)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def single_validation_step(model_: torch.nn.Module,
                           slices_: torch.Tensor,
                           segmentations_: torch.Tensor, has_segmentations: bool,
                           slice_injury_labels: list[torch.Tensor], df_injury_labels: list[torch.Tensor]):
    pred_probas, per_slice_logits, pred_segmentations = model_(slices_)
    # compute metrics for segmentations
    preds = (pred_segmentations > 0).to(torch.float32)
    if has_segmentations:
        loss = focal_loss_segmentation(pred_segmentations, segmentations_, reduce_channels=True)

        tp_pixels = (preds * segmentations_).to(torch.long)
        tn_pixels = ((1 - preds) * (1 - segmentations_)).to(torch.long)
        fp_pixels = (preds * (1 - segmentations_)).to(torch.long)
        fn_pixels = ((1 - preds) * segmentations_).to(torch.long)

        tp_per_class = torch.sum(tp_pixels, dim=tensor_2d3d)
        tn_per_class = torch.sum(tn_pixels, dim=tensor_2d3d)
        fp_per_class = torch.sum(fp_pixels, dim=tensor_2d3d)
        fn_per_class = torch.sum(fn_pixels, dim=tensor_2d3d)

        loss_per_class = focal_loss_segmentation(pred_segmentations, segmentations_, reduce_channels=False)
    else:
        loss = None
        tp_per_class, tn_per_class, fp_per_class, fn_per_class = None, None, None, None
        loss_per_class = None
    segmentation_metrics = (loss, tp_per_class, tn_per_class, fp_per_class, fn_per_class, loss_per_class)
    # compute metrics for injuries
    deep_class_losses, class_losses, pred_classes, per_slice_pred_classes = [], [], [], []
    loss = 0
    for k in range(4):
        is_ternary = UsedLabelManager.is_ternary(k)
        if is_ternary:
            class_loss = ternary_focal_loss_slicepreds(per_slice_logits[k], slice_injury_labels[k])
        else:
            class_loss = binary_focal_loss_slicepreds(per_slice_logits[k], slice_injury_labels[k].unsqueeze(-1))
        loss += (class_loss * 0.25)
        deep_class_losses.append(class_loss.item())

        if is_ternary:
            class_loss = torch.nn.functional.nll_loss(torch.log(torch.clamp(pred_probas[k], min=1e-10)), df_injury_labels[k], reduction="mean")
        else:
            class_loss = torch.nn.functional.binary_cross_entropy(pred_probas[k], df_injury_labels[k].unsqueeze(-1).to(torch.float32), reduction="mean")
        loss += class_loss
        class_losses.append(class_loss.item())

        is_ternary = UsedLabelManager.is_ternary(k)
        if is_ternary:
            pred_classes.append(torch.argmax(pred_probas[k], dim=1))
            per_slice_pred_classes.append(torch.argmax(per_slice_logits[k], dim=1))
        else:
            pred_classes.append((pred_probas[k] > 0.5).squeeze(-1).to(torch.long))
            per_slice_pred_classes.append((per_slice_logits[k] > 0).squeeze(-1).to(torch.long))


    return segmentation_metrics, (loss, deep_class_losses, class_losses, pred_classes, per_slice_pred_classes)

def validation_step():
    for key in val_metrics:
        val_metrics[key].reset()

    # training
    with tqdm.tqdm(total=len(validation_entries[0]) + len(validation_entries[1])) as pbar:
        validation_expert_entries = validation_entries[0]
        validation_remaning_entries = validation_entries[1]
        for k in range(len(validation_expert_entries) + len(validation_remaning_entries)):
            with torch.no_grad():
                if k < len(validation_expert_entries):
                    patient_id = validation_expert_entries[k]  # patient id
                    series_id = str(manager_segmentations.randomly_pick_expert_segmentation(patient_id))
                    seg_folder = manager_segmentations.EXPERT_SEGMENTATION_FOLDER
                    validation_use_segmentations = True
                else:
                    patient_id = validation_remaning_entries[k - len(validation_expert_entries)]
                    series_id = str(manager_folds.randomly_pick_series([patient_id], data_folder="data_hdf5_cropped")[0])
                    seg_folder = -1
                    validation_use_segmentations = False
                any_injury = manager_folds.has_injury(patient_id)
                if any_injury:
                    injury_labels_depth = 5 if use_3d_prediction else 1
                else:
                    injury_labels_depth = -1

                # sample now
                if use_async_sampler:
                    slices, segmentations, injury_labels = image_ROI_sampler_async.load_image(patient_id, series_id,
                                                                                              segmentation_folder=seg_folder,
                                                                                              slices_random=not disable_random_slices,
                                                                                              translate_rotate_augmentation=not disable_rotpos_augmentation,
                                                                                              elastic_augmentation=not disable_elastic_augmentation,
                                                                                              slices=num_slices,
                                                                                              segmentation_region_depth=5 if use_3d_prediction else 1,
                                                                                              injury_labels_depth=injury_labels_depth)
                else:
                    slices, segmentations, injury_labels = image_ROI_sampler.load_image(patient_id, series_id,
                                                                                        segmentation_folder=seg_folder,
                                                                                        slices_random=not disable_random_slices,
                                                                                        translate_rotate_augmentation=not disable_rotpos_augmentation,
                                                                                        elastic_augmentation=not disable_elastic_augmentation,
                                                                                        slices=num_slices,
                                                                                        segmentation_region_depth=5 if use_3d_prediction else 1,
                                                                                        injury_labels_depth=injury_labels_depth)

                # generate per slice labels
                if not any_injury:
                    injury_labels = np.zeros((num_slices, 5), dtype=np.uint8)
                per_slice_class_labels = []
                for k in range(4):
                    if UsedLabelManager.is_ternary(k):
                        labels = torch.tensor(injury_labels[:, k], dtype=torch.long, device=config.device)
                    else:
                        labels = torch.tensor((injury_labels[:, k] > 0), dtype=torch.long, device=config.device)
                    per_slice_class_labels.append(labels)

                # generate labels
                df_injury_labels = []
                for k in range(4):
                    is_ternary = UsedLabelManager.is_ternary(k)
                    df_injury_labels.append(torch.tensor([manager_folds.get_patient_status_single(int(patient_id), k, is_ternary=is_ternary)], dtype=torch.long, device=config.device))

                # evaluate now
                segmentation_metrics, injury_info = single_validation_step(model, slices,
                                                                          segmentations, has_segmentations=validation_use_segmentations,
                                                                          slice_injury_labels=per_slice_class_labels, df_injury_labels=df_injury_labels)
                if validation_use_segmentations:
                    loss, tp_per_class, tn_per_class, fp_per_class, \
                        fn_per_class, loss_per_class = segmentation_metrics
                    loss = loss.item()
                    tp_per_class = tp_per_class.cpu().numpy()
                    tn_per_class = tn_per_class.cpu().numpy()
                    fp_per_class = fp_per_class.cpu().numpy()
                    fn_per_class = fn_per_class.cpu().numpy()
                    loss_per_class = loss_per_class.cpu().numpy()

                    # compute metrics
                    for class_code in range(4):
                        organ_loss_key = MetricKeys.get_segmentation_metric_key_by_class_code(class_code, is_loss=True)
                        organ_key = MetricKeys.get_segmentation_metric_key_by_class_code(class_code, is_loss=False)
                        val_metrics[organ_loss_key].add(loss_per_class[class_code], 1)
                        val_metrics[organ_key].add_direct(tp_per_class[class_code], tn_per_class[class_code],
                                                          fp_per_class[class_code], fn_per_class[class_code])
                    val_metrics[MetricKeys.LOSS].add(loss, 1)

                loss, deep_class_losses, class_losses, pred_classes, per_slice_pred_classes = injury_info
                loss = loss.item()
                # compute metrics
                for class_code in range(4):
                    organ_loss_key = MetricKeys.get_injury_metric_key_by_class_code(class_code,
                                                                                    MetricKeys.METRIC_TYPE_LOSS)
                    organ_injury_key = MetricKeys.get_injury_metric_key_by_class_code(class_code,
                                                                                      MetricKeys.METRIC_TYPE_INJURY)
                    organ_slice_injury_key = MetricKeys.get_injury_metric_key_by_class_code(class_code,
                                                                                            MetricKeys.METRIC_TYPE_SLICE_INJURY)

                    val_metrics[organ_loss_key].add(class_losses[class_code], 1)
                    val_metrics[organ_injury_key].add(pred_classes[class_code], torch.max(per_slice_class_labels[class_code], dim=0, keepdim=True)[0])
                    val_metrics[organ_slice_injury_key].add(per_slice_pred_classes[class_code], per_slice_class_labels[class_code])
                val_metrics[MetricKeys.INJURY_LOSS].add(loss, 1)

            pbar.update(1)

    current_metrics = {}
    for key in train_metrics:
        val_metrics[key].write_to_dict(current_metrics)

    for key in current_metrics:
        val_history[key].append(current_metrics[key])


def create_metrics():
    # create segmentation metrics
    # liver: 0
    train_metrics[MetricKeys.LIVER] = metrics.BinaryMetrics(name="train_liver")
    val_metrics[MetricKeys.LIVER] = metrics.BinaryMetrics(name="val_liver")
    train_metrics[MetricKeys.LIVER_LOSS] = metrics.NumericalMetric(name="train_liver_loss")
    val_metrics[MetricKeys.LIVER_LOSS] = metrics.NumericalMetric(name="val_liver_loss")
    # spleen: 1
    train_metrics[MetricKeys.SPLEEN] = metrics.BinaryMetrics(name="train_spleen")
    val_metrics[MetricKeys.SPLEEN] = metrics.BinaryMetrics(name="val_spleen")
    train_metrics[MetricKeys.SPLEEN_LOSS] = metrics.NumericalMetric(name="train_spleen_loss")
    val_metrics[MetricKeys.SPLEEN_LOSS] = metrics.NumericalMetric(name="val_spleen_loss")
    # kidney: 2
    train_metrics[MetricKeys.KIDNEY] = metrics.BinaryMetrics(name="train_kidney")
    val_metrics[MetricKeys.KIDNEY] = metrics.BinaryMetrics(name="val_kidney")
    train_metrics[MetricKeys.KIDNEY_LOSS] = metrics.NumericalMetric(name="train_kidney_loss")
    val_metrics[MetricKeys.KIDNEY_LOSS] = metrics.NumericalMetric(name="val_kidney_loss")
    # bowel: 3
    train_metrics[MetricKeys.BOWEL] = metrics.BinaryMetrics(name="train_bowel")
    val_metrics[MetricKeys.BOWEL] = metrics.BinaryMetrics(name="val_bowel")
    train_metrics[MetricKeys.BOWEL_LOSS] = metrics.NumericalMetric(name="train_bowel_loss")
    val_metrics[MetricKeys.BOWEL_LOSS] = metrics.NumericalMetric(name="val_bowel_loss")
    # general loss
    train_metrics[MetricKeys.LOSS] = metrics.NumericalMetric(name="train_loss")
    val_metrics[MetricKeys.LOSS] = metrics.NumericalMetric(name="val_loss")

    # create injury metrics
    # liver: 0
    if UsedLabelManager.get_liver_level() == 1:
        train_metrics[MetricKeys.LIVER_INJURY] = metrics.BinaryMetrics(name="train_liver_injury")
        val_metrics[MetricKeys.LIVER_INJURY] = metrics.BinaryMetrics(name="val_liver_injury")
        train_metrics[MetricKeys.LIVER_SLICE_INJURY] = metrics.BinaryMetrics(name="train_liver_slice_injury")
        val_metrics[MetricKeys.LIVER_SLICE_INJURY] = metrics.BinaryMetrics(name="val_liver_slice_injury")
    else:
        train_metrics[MetricKeys.LIVER_INJURY] = metrics.TernaryMetrics(name="train_liver_injury")
        val_metrics[MetricKeys.LIVER_INJURY] = metrics.TernaryMetrics(name="val_liver_injury")
        train_metrics[MetricKeys.LIVER_SLICE_INJURY] = metrics.TernaryMetrics(name="train_liver_slice_injury")
        val_metrics[MetricKeys.LIVER_SLICE_INJURY] = metrics.TernaryMetrics(name="val_liver_slice_injury")
    train_metrics[MetricKeys.LIVER_INJURY_LOSS] = metrics.NumericalMetric(name="train_liver_injury_loss")
    val_metrics[MetricKeys.LIVER_INJURY_LOSS] = metrics.NumericalMetric(name="val_liver_injury_loss")
    # spleen: 1
    if UsedLabelManager.get_spleen_level() == 1:
        train_metrics[MetricKeys.SPLEEN_INJURY] = metrics.BinaryMetrics(name="train_spleen_injury")
        val_metrics[MetricKeys.SPLEEN_INJURY] = metrics.BinaryMetrics(name="val_spleen_injury")
        train_metrics[MetricKeys.SPLEEN_SLICE_INJURY] = metrics.BinaryMetrics(name="train_spleen_slice_injury")
        val_metrics[MetricKeys.SPLEEN_SLICE_INJURY] = metrics.BinaryMetrics(name="val_spleen_slice_injury")
    else:
        train_metrics[MetricKeys.SPLEEN_INJURY] = metrics.TernaryMetrics(name="train_spleen_injury")
        val_metrics[MetricKeys.SPLEEN_INJURY] = metrics.TernaryMetrics(name="val_spleen_injury")
        train_metrics[MetricKeys.SPLEEN_SLICE_INJURY] = metrics.TernaryMetrics(name="train_spleen_slice_injury")
        val_metrics[MetricKeys.SPLEEN_SLICE_INJURY] = metrics.TernaryMetrics(name="val_spleen_slice_injury")
    train_metrics[MetricKeys.SPLEEN_INJURY_LOSS] = metrics.NumericalMetric(name="train_spleen_injury_loss")
    val_metrics[MetricKeys.SPLEEN_INJURY_LOSS] = metrics.NumericalMetric(name="val_spleen_injury_loss")
    # kidney: 2
    if UsedLabelManager.get_kidney_level() == 1:
        train_metrics[MetricKeys.KIDNEY_INJURY] = metrics.BinaryMetrics(name="train_kidney_injury")
        val_metrics[MetricKeys.KIDNEY_INJURY] = metrics.BinaryMetrics(name="val_kidney_injury")
        train_metrics[MetricKeys.KIDNEY_SLICE_INJURY] = metrics.BinaryMetrics(name="train_kidney_slice_injury")
        val_metrics[MetricKeys.KIDNEY_SLICE_INJURY] = metrics.BinaryMetrics(name="val_kidney_slice_injury")
    else:
        train_metrics[MetricKeys.KIDNEY_INJURY] = metrics.TernaryMetrics(name="train_kidney_injury")
        val_metrics[MetricKeys.KIDNEY_INJURY] = metrics.TernaryMetrics(name="val_kidney_injury")
        train_metrics[MetricKeys.KIDNEY_SLICE_INJURY] = metrics.TernaryMetrics(name="train_kidney_slice_injury")
        val_metrics[MetricKeys.KIDNEY_SLICE_INJURY] = metrics.TernaryMetrics(name="val_kidney_slice_injury")
    train_metrics[MetricKeys.KIDNEY_INJURY_LOSS] = metrics.NumericalMetric(name="train_kidney_injury_loss")
    val_metrics[MetricKeys.KIDNEY_INJURY_LOSS] = metrics.NumericalMetric(name="val_kidney_injury_loss")
    # bowel: 3
    train_metrics[MetricKeys.BOWEL_INJURY] = metrics.BinaryMetrics(name="train_bowel_injury")
    val_metrics[MetricKeys.BOWEL_INJURY] = metrics.BinaryMetrics(name="val_bowel_injury")
    train_metrics[MetricKeys.BOWEL_SLICE_INJURY] = metrics.BinaryMetrics(name="train_bowel_slice_injury")
    val_metrics[MetricKeys.BOWEL_SLICE_INJURY] = metrics.BinaryMetrics(name="val_bowel_slice_injury")
    train_metrics[MetricKeys.BOWEL_INJURY_LOSS] = metrics.NumericalMetric(name="train_bowel_injury_loss")
    val_metrics[MetricKeys.BOWEL_INJURY_LOSS] = metrics.NumericalMetric(name="val_bowel_injury_loss")
    # general loss
    train_metrics[MetricKeys.INJURY_LOSS] = metrics.NumericalMetric(name="train_injury_loss")
    val_metrics[MetricKeys.INJURY_LOSS] = metrics.NumericalMetric(name="val_injury_loss")

def print_history(metrics_history: collections.defaultdict[str, list]):
    for key in metrics_history:
        if "injury" in key:
            print("{}      {}".format(key, metrics_history[key][-1]))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Train a ROI prediction model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate to use. Default 3e-4.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--extra_nonexpert_segmentation_training_ratio", type=float, default=0.0, help="Ratio of extra non-expert segmentations to use for training. Default 0.0.")
    parser.add_argument("--disable_random_slices", action="store_true", help="Whether to disable random slices. Default False.")
    parser.add_argument("--disable_rotpos_augmentation", action="store_true", help="Whether to disable rotation and translation augmentation. Default False.")
    parser.add_argument("--disable_elastic_augmentation", action="store_true", help="Whether to disable elastic augmentation. Default False.")
    parser.add_argument("--num_slices", type=int, default=15, help="Number of slices to use. Default 15.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--channel_progression", type=int, nargs="+", default=[2, 3, 6, 9, 15, 32, 128, 256, 512, 1024],
                        help="The channels for progression in ResNet backbone.")
    parser.add_argument("--conv_hidden_channels", type=int, default=32, help="The number of hidden channels for the last conv layers. Default 32.")
    parser.add_argument("--hidden3d_blocks", type=int, nargs="+", default=[1, 2, 1, 0, 0, 0],
                        help="Number of hidden 3d blocks for ResNet backbone.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 2, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false", help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--use_3d_prediction", action="store_true", help="Whether or not to predict a 3D region. Default False.")
    parser.add_argument("--positive_weight", type=float, default=5.0, help="The weight for positive samples for segmentation. Default 5.0.")
    parser.add_argument("--use_async_sampler", action="store_true", help="Whether or not to use an asynchronous sampler. Default False.")
    parser.add_argument("--num_extra_steps", type=int, default=0, help="Extra steps of gradient descent before the usual step in an epoch. Default 0.")
    parser.add_argument("--liver", type=int, default=1, help="Which levels of liver labels to use. Must be 1, 2. Default 1.")
    parser.add_argument("--kidney", type=int, default=1, help="Which levels of kidney labels to use. Must be 1, 2. Default 1.")
    parser.add_argument("--spleen", type=int, default=1, help="Which levels of spleen labels to use. Must be 1, 2. Default 1.")
    manager_folds.add_argparse_arguments(parser)
    manager_models.add_argparse_arguments(parser)
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # check entries
    training_entries, validation_entries, train_dset_name, val_dset_name = manager_folds.parse_args(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    print("Training dataset: {}".format(train_dset_name))
    print("Validation dataset: {}".format(val_dset_name))
    training_entries, validation_entries = initialize_training_entries(training_entries, validation_entries, args.extra_nonexpert_segmentation_training_ratio)


    # initialize gpu
    config.parse_args(args)

    # get model directories
    model_dir, prev_model_dir = manager_models.parse_args(args)

    # get which labels to use
    liver = args.liver
    spleen = args.spleen
    kidney = args.kidney
    assert liver in [1, 2], "Liver must be 1, or 2."
    assert spleen in [1, 2], "Spleen must be 1, or 2."
    assert kidney in [1, 2], "Kidney must be 1, or 2."
    UsedLabelManager.levels_used = [liver, spleen, kidney]
    print("Using label levels {} for liver, {} for spleen, and {} for kidney.".format(liver, spleen, kidney))

    # obtain model and training parameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    second_momentum = args.second_momentum
    disable_random_slices = args.disable_random_slices
    disable_rotpos_augmentation = args.disable_rotpos_augmentation
    disable_elastic_augmentation = args.disable_elastic_augmentation
    num_slices = args.num_slices
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    channel_progression = args.channel_progression
    conv_hidden_channels = args.conv_hidden_channels
    hidden3d_blocks = args.hidden3d_blocks
    hidden_blocks = args.hidden_blocks
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    use_3d_prediction = args.use_3d_prediction
    positive_weight = args.positive_weight
    use_async_sampler = args.use_async_sampler
    num_extra_steps = args.num_extra_steps

    print("Epochs: " + str(epochs))
    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Disable random slices: " + str(disable_random_slices))
    print("Disable rotation and translation augmentation: " + str(disable_rotpos_augmentation))
    print("Disable elastic augmentation: " + str(disable_elastic_augmentation))
    print("Number of slices: " + str(num_slices))

    assert type(hidden_blocks) == list, "Blocks must be a list."
    assert type(channel_progression) == list, "Channels must be a list."
    assert type(hidden3d_blocks) == list, "3D Blocks must be a list."
    for k in hidden_blocks:
        assert type(k) == int, "Blocks must be a list of integers."
    for k in channel_progression:
        assert type(k) == int, "Channels must be a list of integers."
    for k in hidden3d_blocks:
        assert type(k) == int, "3D Blocks must be a list of integers."

    print("Hidden channels: " + str(channel_progression))
    print("Conv hidden channels: " + str(conv_hidden_channels))
    print("Hidden blocks: " + str(hidden_blocks))
    print("Hidden 3D blocks: " + str(hidden3d_blocks))
    print("Bottleneck factor: " + str(bottleneck_factor))
    print("Squeeze and excitation: " + str(squeeze_excitation))
    print("Use 3D prediction: " + str(use_3d_prediction))
    print("Positive weight: " + str(positive_weight))

    # Create model and optimizer, and setup 3d or 2d
    backbone = model_3d_patch_resnet.ResNet3DBackbone(in_channels=1,
                                                      channel_progression=channel_progression,
                                                      res_conv3d_blocks=hidden3d_blocks,
                                                      res_conv_blocks=hidden_blocks,
                                                      bottleneck_factor=bottleneck_factor,
                                                      squeeze_excitation=squeeze_excitation,
                                                      return_3d_features=use_3d_prediction)
    if use_3d_prediction:
        deep_channels = backbone.get_deep3d_channels()
        print("Using 3D prediction. Detected deep channels: " + str(deep_channels) + "   Last channel: " + str(
            channel_progression[-1]))
        model = model_3d_patch_resnet.SupervisedAttentionClassifier3D(backbone=backbone, backbone_first_channels=deep_channels[0],
                                                        backbone_mid_channels=deep_channels[1], backbone_last_channels=channel_progression[-1],
                                                        backbone_feature_width=18, backbone_feature_height=16, conv_hidden_channels=conv_hidden_channels,
                                                        classification_levels=UsedLabelManager.levels_used + [1], reduction="union")
        tensor_2d3d = (0, 2, 3, 4)
        bowel_mask_tensor = torch.tensor([1, 1, 1, 0], dtype=torch.float32, device=config.device) \
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        model = model_3d_patch_resnet.SupervisedAttentionClassifier(backbone=backbone, backbone_out_channels=channel_progression[-1],
                                                                    conv_hidden_channels=conv_hidden_channels, classification_levels=UsedLabelManager.levels_used + [1],
                                                                    reduction="union")
        tensor_2d3d = (0, 2, 3)
        bowel_mask_tensor = torch.tensor([1, 1, 1, 0], dtype=torch.float32, device=config.device) \
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    model = model.to(config.device)

    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Optimizer: " + optimizer_type)
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(momentum, second_momentum))
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        print("Invalid optimizer. The available options are: adam, sgd.")
        exit(1)

    # Load previous model checkpoint if available
    if prev_model_dir is not None:
        model_checkpoint_path = os.path.join(prev_model_dir, "model.pt")
        optimizer_checkpoint_path = os.path.join(prev_model_dir, "optimizer.pt")

        model.load_state_dict(torch.load(model_checkpoint_path))
        optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))

        for g in optimizer.param_groups:
            g["lr"] = learning_rate
            if optimizer_type == "adam":
                g["betas"] = (momentum, second_momentum)
            elif optimizer_type == "sgd":
                g["momentum"] = momentum

    model_config = {
        "model": "Organ injury predictions with attention using ROI classification as deep supervision",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "second_momentum": second_momentum,
        "extra_nonexpert_segmentation_training_ratio": args.extra_nonexpert_segmentation_training_ratio,
        "disable_random_slices": disable_random_slices,
        "disable_rotpos_augmentation": disable_rotpos_augmentation,
        "disable_elastic_augmentation": disable_elastic_augmentation,
        "num_slices": num_slices,
        "optimizer": optimizer_type,
        "epochs_per_save": epochs_per_save,
        "channel_progression": channel_progression,
        "hidden_blocks": hidden_blocks,
        "hidden3d_blocks": hidden3d_blocks,
        "bottleneck_factor": bottleneck_factor,
        "squeeze_excitation": squeeze_excitation,
        "use_3d_prediction": use_3d_prediction,
        "positive_weight": positive_weight,
        "use_async_sampler": use_async_sampler,
        "num_extra_steps": num_extra_steps,
        "liver": liver,
        "kidney": kidney,
        "spleen": spleen,
        "train_dataset": train_dset_name,
        "val_dataset": val_dset_name,
        "training_script": "training_ROI_classifier.py",
    }

    # Save the model config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # Create the metrics
    train_history = collections.defaultdict(list)
    train_metrics = {}
    val_history = collections.defaultdict(list)
    val_metrics = {}
    create_metrics()

    # Compile
    single_training_step_segmentation_compile = torch.compile(single_training_step_segmentation)
    single_training_step_injury_compile = torch.compile(single_training_step_injury)

    # Initialize the async sampler if necessary
    if use_async_sampler:
        print("Initializing async sampler....")
        image_ROI_sampler_async.initialize_async_ROI_sampler(use_3d=use_3d_prediction, name=train_dset_name)

    # Start training loop
    print("Training for {} epochs......".format(epochs))
    memory_logger = logging_memory_utils.obtain_memory_logger(model_dir)

    try:
        for epoch in range(epochs):
            memory_logger.log("Epoch {}".format(epoch))
            print("------------------------------------ Epoch {} ------------------------------------".format(epoch))
            model.train()
            print("Running {} extra steps of gradient descent.".format(num_extra_steps))
            for k in range(num_extra_steps):
                training_step(record=False)
                torch.save(model.state_dict(), os.path.join(model_dir, "model_{}_substep{}.pt".format(epoch, k)))
                torch.save(optimizer.state_dict(),
                           os.path.join(model_dir, "optimizer_{}_substep{}.pt".format(epoch, k)))

            print("Running the usual step of gradient descent.")
            training_step(record=True)

            # switch model to eval mode, and reset all running stats for batchnorm layers
            model.eval()
            with torch.no_grad():
                validation_step()

            print()
            print_history(train_history)
            print_history(val_history)
            # save metrics
            train_df = pd.DataFrame(train_history)
            val_df = pd.DataFrame(val_history)
            train_df.to_csv(os.path.join(model_dir, "train_metrics.csv"), index=True)
            val_df.to_csv(os.path.join(model_dir, "val_metrics.csv"), index=True)

            if epoch % epochs_per_save == 0:
                torch.save(model.state_dict(), os.path.join(model_dir, "model_{}.pt".format(epoch)))
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_{}.pt".format(epoch)))

        print("Training complete! Saving and finalizing...")
    except KeyboardInterrupt:
        print("Training interrupted! Saving and finalizing...")
    except Exception as e:
        print("Training interrupted due to exception! Saving and finalizing...")
        traceback.print_exc()
    # save model
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))

    # save metrics
    if len(train_history) > 0:
        train_df = pd.DataFrame(train_history)
        val_df = pd.DataFrame(val_history)
        train_df.to_csv(os.path.join(model_dir, "train_metrics.csv"), index=True)
        val_df.to_csv(os.path.join(model_dir, "val_metrics.csv"), index=True)

    memory_logger.close()

    if use_async_sampler:
        image_ROI_sampler_async.clean_and_destroy_ROI_sampler()
