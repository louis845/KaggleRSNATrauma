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
    LOSS = "loss"
    LIVER = "liver"
    LIVER_LOSS = "liver_loss"
    SPLEEN = "spleen"
    SPLEEN_LOSS = "spleen_loss"
    KIDNEY = "kidney"
    KIDNEY_LOSS = "kidney_loss"
    BOWEL = "bowel"
    BOWEL_LOSS = "bowel_loss"

    @staticmethod
    def get_metric_key_by_class_code(class_code: int, is_loss: bool):
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


def get_representing_shuffle_entry(patient_id: int):
    return {"patient_id": patient_id}


def get_patient_id_from_shuffle_entry(shuffle_entry: dict) -> int:
    return shuffle_entry["patient_id"]


bowel_mask_tensor = None  # to be initialized with tensor_2d3d


# focal loss with exponent 2
def focal_loss(output: torch.Tensor, target: torch.Tensor, reduce_channels=True, include_bowel=True):
    with torch.no_grad():
        extra_weight = (target * (positive_weight - 1) + # weights for positive samples
                         torch.amax((1 - target), dim=tensor_lastdims, keepdim=True) * (no_labels_extra_weight - 1)) # weights for negative samples
    # logsumexp trick for numerical stability
    binary_ce = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction="none") * (1 + extra_weight)
    if not include_bowel:
        binary_ce = binary_ce * bowel_mask_tensor
    if reduce_channels:
        return torch.mean(((target - torch.sigmoid(output)) ** 2) * binary_ce)
    else:
        return torch.mean(((target - torch.sigmoid(output)) ** 2) * binary_ce, dim=tensor_2d3d)


def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                         slices_: torch.Tensor, segmentations_: torch.Tensor, is_expert: bool):
    optimizer_.zero_grad()
    pred_segmentations = model_(slices_)
    with torch.no_grad():
        preds = (pred_segmentations > 0).to(torch.float32)
    # we include the bowel only for expert segmentations, and penalize non-expert segmentations only half as much
    loss = focal_loss(pred_segmentations, segmentations_, reduce_channels=True, include_bowel=is_expert)
    if not is_expert:
        loss = loss * 0.5
    loss.backward()
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

        loss_per_class = focal_loss(pred_segmentations, segmentations_, reduce_channels=False)

    return loss, tp_per_class, tn_per_class, fp_per_class, fn_per_class, loss_per_class


def single_validation_step(model_: torch.nn.Module,
                           slices_: torch.Tensor,
                           segmentations_: torch.Tensor):
    pred_segmentations = model_(slices_)
    preds = (pred_segmentations > 0).to(torch.float32)
    loss = focal_loss(pred_segmentations, segmentations_, reduce_channels=True)

    tp_pixels = (preds * segmentations_).to(torch.long)
    tn_pixels = ((1 - preds) * (1 - segmentations_)).to(torch.long)
    fp_pixels = (preds * (1 - segmentations_)).to(torch.long)
    fn_pixels = ((1 - preds) * segmentations_).to(torch.long)

    tp_per_class = torch.sum(tp_pixels, dim=tensor_2d3d)
    tn_per_class = torch.sum(tn_pixels, dim=tensor_2d3d)
    fp_per_class = torch.sum(fp_pixels, dim=tensor_2d3d)
    fn_per_class = torch.sum(fn_pixels, dim=tensor_2d3d)

    loss_per_class = focal_loss(pred_segmentations, segmentations_, reduce_channels=False)

    return loss, tp_per_class, tn_per_class, fp_per_class, fn_per_class, loss_per_class


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
            patient_id = get_patient_id_from_shuffle_entry(training_entries[shuffle_indices[trained]])  # patient id
            if manager_segmentations.patient_has_expert_segmentations(patient_id):
                series_id = str(manager_segmentations.randomly_pick_expert_segmentation(patient_id))
                is_expert = True
                seg_folder = manager_segmentations.EXPERT_SEGMENTATION_FOLDER
            else:
                series_id = str(manager_segmentations.randomly_pick_TSM_segmentation(patient_id))
                is_expert = False
                seg_folder = manager_segmentations.TSM_SEGMENTATION_FOLDER
            if use_async_sampler:
                slices, segmentations = image_ROI_sampler_async.load_image(patient_id, series_id,
                                                                           segmentation_folder=seg_folder,
                                                                           slices_random=not disable_random_slices,
                                                                           translate_rotate_augmentation=not disable_rotpos_augmentation,
                                                                           elastic_augmentation=not disable_elastic_augmentation,
                                                                           boundary_augmentation=use_boundary_augmentation,
                                                                           slices=num_slices,
                                                                           segmentation_region_depth=5 if use_3d_prediction else 1,)
            else:
                slices, segmentations = image_ROI_sampler.load_image(patient_id, series_id,
                                                                     segmentation_folder=seg_folder,
                                                                     slices_random=not disable_random_slices,
                                                                     translate_rotate_augmentation=not disable_rotpos_augmentation,
                                                                     elastic_augmentation=not disable_elastic_augmentation,
                                                                     boundary_augmentation=use_boundary_augmentation,
                                                                     slices=num_slices,
                                                                     segmentation_region_depth=5 if use_3d_prediction else 1)

            loss, tp_per_class, tn_per_class, fp_per_class, \
                fn_per_class, loss_per_class = single_training_step_compile(model, optimizer, slices, segmentations,
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
                    organ_loss_key = MetricKeys.get_metric_key_by_class_code(class_code, is_loss=True)
                    organ_key = MetricKeys.get_metric_key_by_class_code(class_code, is_loss=False)
                    train_metrics[organ_loss_key].add(loss_per_class[class_code], 1)
                    train_metrics[organ_key].add_direct(tp_per_class[class_code], tn_per_class[class_code],
                                                        fp_per_class[class_code], fn_per_class[class_code])
                train_metrics[MetricKeys.LOSS].add(loss, 1)

            trained += 1
            pbar.update(1)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])


def validation_step():
    for key in val_metrics:
        val_metrics[key].reset()

    # training
    validated = 0
    with tqdm.tqdm(total=len(validation_entries)) as pbar:
        while validated < len(validation_entries):
            with torch.no_grad():
                patient_id = validation_entries[validated]  # patient id
                if manager_segmentations.patient_has_expert_segmentations(patient_id):
                    series_id = str(manager_segmentations.randomly_pick_expert_segmentation(patient_id))
                    seg_folder = manager_segmentations.EXPERT_SEGMENTATION_FOLDER
                else:
                    series_id = str(manager_segmentations.randomly_pick_TSM_segmentation(patient_id))
                    seg_folder = manager_segmentations.TSM_SEGMENTATION_FOLDER
                if use_async_sampler:
                    slices, segmentations = image_ROI_sampler_async.load_image(patient_id, series_id,
                                                                               segmentation_folder=seg_folder,
                                                                               slices_random=False,
                                                                               translate_rotate_augmentation=False,
                                                                               elastic_augmentation=False,
                                                                               boundary_augmentation=False,
                                                                               slices=num_slices,
                                                                               segmentation_region_depth=5 if use_3d_prediction else 1)
                else:
                    slices, segmentations = image_ROI_sampler.load_image(patient_id, series_id,
                                                                         segmentation_folder=seg_folder,
                                                                         slices_random=False,
                                                                         translate_rotate_augmentation=False,
                                                                         elastic_augmentation=False,
                                                                         boundary_augmentation=False,
                                                                         slices=num_slices,
                                                                         segmentation_region_depth=5 if use_3d_prediction else 1)

                loss, tp_per_class, tn_per_class, fp_per_class, \
                    fn_per_class, loss_per_class = single_validation_step(model, slices, segmentations)
                loss = loss.item()
                tp_per_class = tp_per_class.cpu().numpy()
                tn_per_class = tn_per_class.cpu().numpy()
                fp_per_class = fp_per_class.cpu().numpy()
                fn_per_class = fn_per_class.cpu().numpy()
                loss_per_class = loss_per_class.cpu().numpy()

                # compute metrics
                for class_code in range(4):
                    organ_loss_key = MetricKeys.get_metric_key_by_class_code(class_code, is_loss=True)
                    organ_key = MetricKeys.get_metric_key_by_class_code(class_code, is_loss=False)
                    val_metrics[organ_loss_key].add(loss_per_class[class_code], 1)
                    val_metrics[organ_key].add_direct(tp_per_class[class_code], tn_per_class[class_code],
                                                      fp_per_class[class_code], fn_per_class[class_code])
                val_metrics[MetricKeys.LOSS].add(loss, 1)

            validated += 1
            pbar.update(1)

    current_metrics = {}
    for key in train_metrics:
        val_metrics[key].write_to_dict(current_metrics)

    for key in current_metrics:
        val_history[key].append(current_metrics[key])


def print_history(metrics_history: collections.defaultdict[str, list]):
    for key in metrics_history:
        print("{}      {}".format(key, metrics_history[key][-1]))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Train a ROI prediction model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate to use. Default 3e-4.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999,
                        help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--num_extra_nonexpert_training", type=int, default=0,
                        help="Number of extra non-expert segmentations to use for training. Default 0.")
    parser.add_argument("--disable_random_slices", action="store_true",
                        help="Whether to disable random slices. Default False.")
    parser.add_argument("--disable_rotpos_augmentation", action="store_true",
                        help="Whether to disable rotation and translation augmentation. Default False.")
    parser.add_argument("--disable_elastic_augmentation", action="store_true",
                        help="Whether to disable elastic augmentation. Default False.")
    parser.add_argument("--use_boundary_augmentation", action="store_true",
                        help="Whether to use boundary augmentation. Default False.")
    parser.add_argument("--num_slices", type=int, default=15, help="Number of slices to use. Default 15.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--channel_progression", type=int, nargs="+", default=[2, 3, 6, 9, 15, 32, 128, 256, 512, 1024],
                        help="The channels for progression in ResNet backbone.")
    parser.add_argument("--hidden3d_blocks", type=int, nargs="+", default=[1, 2, 1, 0, 0, 0],
                        help="Number of hidden 3d blocks for ResNet backbone.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 2, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--bottleneck_factor", type=int, default=4,
                        help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false",
                        help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--use_3d_prediction", action="store_true",
                        help="Whether or not to predict a 3D region. Default False.")
    parser.add_argument("--positive_weight", type=float, default=8.0,
                        help="The weight for positive samples. Default 5.0.")
    parser.add_argument("--no_labels_extra_weight", type=float, default=3.0,
                        help="The weight for slices with no labels. Default 2.0.")
    parser.add_argument("--use_async_sampler", action="store_true",
                        help="Whether or not to use an asynchronous sampler. Default False.")
    parser.add_argument("--num_extra_steps", type=int, default=0,
                        help="Extra steps of gradient descent before the usual step in an epoch. Default 0.")
    manager_folds.add_argparse_arguments(parser)
    manager_models.add_argparse_arguments(parser)
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # check entries
    training_entries, validation_entries, train_dset_name, val_dset_name = manager_folds.parse_args(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    training_entries = np.array(training_entries)
    validation_entries = np.array(validation_entries)
    print("Training dataset: {}".format(train_dset_name))
    print("Validation dataset: {}".format(val_dset_name))

    for patient_id in training_entries:
        assert manager_segmentations.patient_has_expert_segmentations(
            patient_id) or manager_segmentations.patient_has_TSM_segmentations(
            patient_id), "Patient {} has no expert or TSM segmentations.".format(patient_id)
    for patient_id in validation_entries:
        assert manager_segmentations.patient_has_expert_segmentations(
            patient_id) or manager_segmentations.patient_has_TSM_segmentations(
            patient_id), "Patient {} has no expert or TSM segmentations.".format(patient_id)
    if args.num_extra_nonexpert_training > 0:
        print("Using {} extra non-expert segmentations for training.".format(args.num_extra_nonexpert_training))
        expert_patients = manager_segmentations.get_patients_with_expert_segmentation()
        extra_entries = [int(x) for x in manager_segmentations.get_patients_with_TSM_segmentation() if
                            str(x) not in expert_patients]
        training_entries = [int(x) for x in training_entries if str(x) in expert_patients]
        if len(set(extra_entries).intersection(set([int(x) for x in validation_entries]))) > 0:
            original_length = len(extra_entries)
            extra_entries = set(extra_entries).difference(set([int(x) for x in validation_entries]))
            print("Removed {} extra non-expert segmentations that were in the validation set.".format(original_length -
                                                                                                      len(extra_entries)))
            print("Final number of extra non-expert segmentations: {}".format(len(extra_entries)))
        assert len(set(extra_entries).intersection(set([int(x) for x in validation_entries]))) == 0, \
            "Some validation patients have TSM (non-expert) segmentations."
        validation_entries = [x for x in validation_entries if str(x) in expert_patients]
        training_entries = training_shuffle_utils.BiasedShuffleInfo(shuffle_info=[get_representing_shuffle_entry(int(x)) for
                                                                            x in training_entries],
                                                 shuffle_info_extra=[get_representing_shuffle_entry(int(x)) for
                                                                            x in extra_entries],
                                                 extra_shuffle_samples=args.num_extra_nonexpert_training)
    else:
        print("Not using any extra segmentations.")
        training_entries = training_shuffle_utils.ShuffleInfo(shuffle_info=[get_representing_shuffle_entry(int(x)) for
                                                                            x in training_entries])

    # initialize gpu
    config.parse_args(args)

    # get model directories
    model_dir, prev_model_dir = manager_models.parse_args(args)

    # obtain model and training parameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    second_momentum = args.second_momentum
    disable_random_slices = args.disable_random_slices
    disable_rotpos_augmentation = args.disable_rotpos_augmentation
    disable_elastic_augmentation = args.disable_elastic_augmentation
    use_boundary_augmentation = args.use_boundary_augmentation
    num_slices = args.num_slices
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    channel_progression = args.channel_progression
    hidden3d_blocks = args.hidden3d_blocks
    hidden_blocks = args.hidden_blocks
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    use_3d_prediction = args.use_3d_prediction
    positive_weight = args.positive_weight
    no_labels_extra_weight = args.no_labels_extra_weight
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
    print("Hidden blocks: " + str(hidden_blocks))
    print("Hidden 3D blocks: " + str(hidden3d_blocks))
    print("Bottleneck factor: " + str(bottleneck_factor))
    print("Squeeze and excitation: " + str(squeeze_excitation))
    print("Use 3D prediction: " + str(use_3d_prediction))
    print("Positive weight: " + str(positive_weight))
    print("No labels extra weight: " + str(no_labels_extra_weight))

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
        model = model_3d_patch_resnet.NeighborhoodROINet(backbone=backbone, first_channels=deep_channels[0],
                                                         mid_channels=deep_channels[1],
                                                         last_channels=channel_progression[-1],
                                                         feature_width=18, feature_height=16)
        tensor_2d3d = (0, 2, 3, 4)
        tensor_lastdims = (2, 3, 4)
        bowel_mask_tensor = torch.tensor([1, 1, 1, 0], dtype=torch.float32, device=config.device) \
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        model = model_3d_patch_resnet.LocalizedROINet(backbone=backbone, num_channels=channel_progression[-1])
        tensor_2d3d = (0, 2, 3)
        tensor_lastdims = (2, 3)
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
        "model": "Organ ROI classifier",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "second_momentum": second_momentum,
        "num_extra_nonexpert_training": args.num_extra_nonexpert_training,
        "disable_random_slices": disable_random_slices,
        "disable_rotpos_augmentation": disable_rotpos_augmentation,
        "disable_elastic_augmentation": disable_elastic_augmentation,
        "use_boundary_augmentation": use_boundary_augmentation,
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
        "no_labels_extra_weight": no_labels_extra_weight,
        "use_async_sampler": use_async_sampler,
        "num_extra_steps": num_extra_steps,
        "train_dataset": train_dset_name,
        "val_dataset": val_dset_name,
        "training_script": "training_ROI_preds.py",
    }

    # Save the model config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # Create the metrics
    train_history = collections.defaultdict(list)
    train_metrics = {}
    val_history = collections.defaultdict(list)
    val_metrics = {}
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

    # Compile
    single_training_step_compile = torch.compile(single_training_step)

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
