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
import manager_segmentations
import manager_stage1_results
import model_resnet_old
import metrics
import image_organ_sampler
import training_shuffle_utils


def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
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
            if k == 4:
                loss += (class_loss)
            deep_class_losses.append(class_loss.item())

        if is_ternary:
            class_loss = ternary_loss(pred_probas[k], torch.max(injury_labels[k], dim=0, keepdim=True)[0])
        else:
            class_loss = binary_loss(pred_probas[k], torch.max(injury_labels[k], dim=0, keepdim=True)[0].unsqueeze(0).to(torch.float32))
        if k != 4:
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
                if final_predictions_logits:
                    pred_classes.append((pred_probas[k] > 0.0).squeeze(-1).to(torch.long))
                else:
                    pred_classes.append((pred_probas[k] > 0.5).squeeze(-1).to(torch.long))
                per_slice_pred_classes.append((per_slice_logits[k] > 0).squeeze(-1).to(torch.long))

    return loss, deep_class_losses, class_losses, pred_classes, per_slice_pred_classes

def training_step(record: bool):
    if record:
        for key in train_metrics:
            train_metrics[key].reset()

    # shuffle
    shuffle_indices = np.random.permutation(len(training_entries))

    # training
    trained = 0
    with tqdm.tqdm(total=len(shuffle_indices)) as pbar:
        while trained < len(shuffle_indices):
            length = min(len(shuffle_indices) - trained, batch_size)
            batch_entries = training_entries[shuffle_indices[trained:trained + length]]
            # prepare options for image sampler
            series1, series2 = train_stage1_results.get_dual_series(batch_entries, organ_id=organ_id)

            # sample now
            image_batch = image_organ_sampler.load_image(batch_entries,
                                           series1,
                                           organ_id, 360, 360,
                                           train_stage1_results,
                                           volume_depth,
                                           translate_rotate_augmentation=not disable_rotpos_augmentation,
                                           elastic_augmentation=not disable_elastic_augmentation)

            # do training now
            loss, preds = single_training_step(model, optimizer, image_batch, injury_labels, deep_guidance)

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


            trained += 1
            pbar.update(1)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def print_history(metrics_history):
    for key in metrics_history:
        print("{}      {}".format(key, metrics_history[key][-1]))

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Train a injury prediction model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate to use. Default 3e-4.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--disable_rotpos_augmentation", action="store_true", help="Whether to disable rotation and translation augmentation. Default False.")
    parser.add_argument("--disable_elastic_augmentation", action="store_true", help="Whether to disable elastic augmentation. Default False.")
    parser.add_argument("--batch_size", type=int, default=9, help="Batch size. Default 9")
    parser.add_argument("--volume_depth", type=int, default=9, help="Number of slices to use per volume. Default 9.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels. Default 64.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false", help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--num_extra_steps", type=int, default=0, help="Extra steps of gradient descent before the usual step in an epoch. Default 0.")
    parser.add_argument("--organ", type=str, help="Which organ to train on. Default liver. Available options: liver, spleen, kidney", required=True)
    manager_folds.add_argparse_arguments(parser)
    manager_models.add_argparse_arguments(parser)
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # get organ
    organ = args.organ
    assert args.organ in ["liver", "spleen", "kidney"], "Organ must be liver, spleen, or kidney."
    organ_id = manager_stage1_results.Stage1ResultsManager.organs.index(organ)

    # check entries
    training_entries, validation_entries, train_dset_name, val_dset_name = manager_folds.parse_args(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    print("Training dataset: {}".format(train_dset_name))
    print("Validation dataset: {}".format(val_dset_name))

    train_stage1_results = manager_stage1_results.Stage1ResultsManager(train_dset_name)
    val_stage1_results = manager_stage1_results.Stage1ResultsManager(val_dset_name)
    train_stage1_results.validate_patient_ids_contained(training_entries)
    val_stage1_results.validate_patient_ids_contained(validation_entries)

    training_entries = train_stage1_results.restrict_patient_ids_to_good_series(training_entries)
    validation_entries = val_stage1_results.restrict_patient_ids_to_good_series(validation_entries)

    training_entries = train_stage1_results.restrict_patient_ids_to_organs(training_entries, organ_id)
    validation_entries = val_stage1_results.restrict_patient_ids_to_organs(validation_entries, organ_id)

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
    batch_size = args.batch_size
    volume_depth = args.volume_depth
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    num_extra_steps = args.num_extra_steps

    print("Epochs: " + str(epochs))
    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Disable random slices: " + str(disable_random_slices))
    print("Disable rotation and translation augmentation: " + str(disable_rotpos_augmentation))
    print("Disable elastic augmentation: " + str(disable_elastic_augmentation))
    print("Batch size: " + str(batch_size))

    assert type(hidden_blocks) == list, "Blocks must be a list."
    for k in hidden_blocks:
        assert type(k) == int, "Blocks must be a list of integers."
    assert hidden_channels % 4 == 0, "Hidden channels must be divisible by 4."

    backbone = model_resnet_old.ResNetBackbone(volume_depth * 2, hidden_channels // bottleneck_factor, use_batch_norm=True,
                                               use_res_conv=True, pyr_height=4, res_conv_blocks=hidden_blocks,
                                               bottleneck_expansion=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                                               use_initial_conv=True)
    model = model_resnet_old.ResNetClassifier(backbone, hidden_channels * (2 ** (len(hidden_blocks) - 1)), out_classes=3)
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
        "batch_size": batch_size,
        "volume_depth": volume_depth,
        "optimizer": optimizer_type,
        "hidden_blocks": hidden_blocks,
        "hidden_channels": hidden_channels,
        "bottleneck_factor": bottleneck_factor,
        "squeeze_excitation": squeeze_excitation,
        "num_extra_steps": num_extra_steps,
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
    train_metrics["loss"] = metrics.NumericalMetric("train_loss")
    train_metrics["metric"] = metrics.TernaryMetrics("train_metric")
    val_metrics["loss"] = metrics.NumericalMetric("val_loss")
    val_metrics["metric"] = metrics.TernaryMetrics("val_metric")

    # Compile
    single_training_step_compile = torch.compile(single_training_step)

    # Initialize the async sampler if necessary
    """
    if use_async_sampler:
        print("Initializing async sampler....")
        image_ROI_sampler_async.initialize_async_ROI_sampler(use_3d=use_3d_prediction, name=train_dset_name)"""

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

    """if use_async_sampler:
        image_ROI_sampler_async.clean_and_destroy_ROI_sampler()"""
