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
import model_3d_predictor_resnet
import model_resnet_old
import metrics
import image_organ_sampler
import training_shuffle_utils


def get_labels(batch_entries):
    stack = []
    for patient_id in batch_entries:
        if organ_id == 0: # liver
            status = manager_folds.get_liver_status_single(int(patient_id))
        elif organ_id == 1: # spleen
            status = manager_folds.get_spleen_status_single(int(patient_id))
        elif organ_id == 2: # kidney
            status = manager_folds.get_kidney_status_single(int(patient_id))
        assert status.shape == (3,)
        assert isinstance(status, np.ndarray)
        stack.append(status)
    return np.stack(stack, axis=0).astype(np.float32)

weights_tensor = None # to be initialized
def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            image_batch: torch.Tensor, labels_batch: torch.Tensor):
    with torch.no_grad():
        labels_batch_amax = torch.argmax(labels_batch, dim=-1)
        weights = torch.sum(labels_batch * weights_tensor, dim=-1)
    optimizer_.zero_grad()
    pred_logits = model_(image_batch)
    loss = torch.nn.functional.cross_entropy(pred_logits, labels_batch_amax, reduction="none")
    loss = torch.sum(loss * weights)
    loss.backward()
    optimizer_.step()

    with torch.no_grad():
        preds = torch.argmax(pred_logits, dim=-1)

    return loss.item(), preds, weights.cpu().numpy()

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
            batch_entries = training_entries[shuffle_indices[trained:trained + length]] # patient ids
            # prepare options for image sampler
            series1, series2 = train_stage1_results.get_dual_series(batch_entries, organ_id=organ_id)

            # sample now
            image_batch1 = image_organ_sampler.load_image(batch_entries,
                                           series1,
                                           organ_id, organ_size[0], organ_size[1],
                                           train_stage1_results,
                                           volume_depth,
                                           translate_rotate_augmentation=not disable_rotpos_augmentation,
                                           elastic_augmentation=not disable_elastic_augmentation)
            image_batch2 = image_organ_sampler.load_image(batch_entries,
                                           series2,
                                           organ_id, organ_size[0], organ_size[1],
                                           train_stage1_results,
                                           volume_depth,
                                           translate_rotate_augmentation=not disable_rotpos_augmentation,
                                           elastic_augmentation=not disable_elastic_augmentation)
            image_batch = torch.cat([image_batch1, image_batch2], dim=2)
            if using_resnet:
                image_batch = image_batch.squeeze(1)  # (N, 1, 2 * D, H, W) -> (N, 2 * D, H, W)
            labels_batch = torch.tensor(get_labels(batch_entries), device=config.device, dtype=torch.float32)

            # do training now
            loss, preds, weights = single_training_step(model, optimizer, image_batch, labels_batch)

            # record
            if record:
                with torch.no_grad():
                    train_metrics["loss"].add(loss, sum(list(weights)))
                    train_metrics["metric"].add(preds, torch.argmax(labels_batch, dim=-1))

            trained += length
            pbar.update(length)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def single_validation_step(model_: torch.nn.Module, image_batch: torch.Tensor, labels_batch: torch.Tensor):
    labels_batch_amax = torch.argmax(labels_batch, dim=-1)
    weights = torch.sum(labels_batch * weights_tensor, dim=-1)
    pred_logits = model_(image_batch)
    loss = torch.nn.functional.cross_entropy(pred_logits, labels_batch_amax, reduction="none")
    loss = torch.sum(loss * weights)
    preds = torch.argmax(pred_logits, dim=-1)

    return loss.item(), preds, weights.cpu().numpy()

def validation_step():
    for key in val_metrics:
        val_metrics[key].reset()

    # validation
    validated = 0
    with tqdm.tqdm(total=len(validation_entries)) as pbar:
        while validated < len(validation_entries):
            length = min(len(validation_entries) - validated, batch_size)
            batch_entries = validation_entries[validated:validated + length] # patient ids
            # prepare options for image sampler
            series1, series2 = val_stage1_results.get_dual_series(batch_entries, organ_id=organ_id)

            # sample now
            image_batch1 = image_organ_sampler.load_image(batch_entries,
                                           series1,
                                           organ_id, organ_size[0], organ_size[1],
                                           val_stage1_results,
                                           volume_depth,
                                           translate_rotate_augmentation=False,
                                           elastic_augmentation=False)
            image_batch2 = image_organ_sampler.load_image(batch_entries,
                                           series2,
                                           organ_id, organ_size[0], organ_size[1],
                                           val_stage1_results,
                                           volume_depth,
                                           translate_rotate_augmentation=False,
                                           elastic_augmentation=False)
            with torch.no_grad():
                image_batch = torch.cat([image_batch1, image_batch2], dim=2)
                if using_resnet:
                    image_batch = image_batch.squeeze(1) # (N, 1, 2 * D, H, W) -> (N, 2 * D, H, W)
                labels_batch = torch.tensor(get_labels(batch_entries), device=config.device, dtype=torch.float32)

                # do validation now
                loss, preds, weights = single_validation_step(model, image_batch, labels_batch)

                val_metrics["loss"].add(loss, sum(list(weights)))
                val_metrics["metric"].add(preds, torch.argmax(labels_batch, dim=-1))

            validated += length
            pbar.update(length)

    current_metrics = {}
    for key in val_metrics:
        val_metrics[key].write_to_dict(current_metrics)

    for key in current_metrics:
        val_history[key].append(current_metrics[key])

def print_history(metrics_history):
    for key in metrics_history:
        print("{}      {}".format(key, metrics_history[key][-1]))

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    sizes = [ # sizes in (H, W)
        (352, 448), # 0: liver
        (320, 416), # 1: spleen
        (224, 352) # 2: kidney
    ]

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
    parser.add_argument("--hidden_channels", type=int, default=None, help="Number of hidden channels. Default None.")
    parser.add_argument("--channel_progression", type=int, nargs="+", default=None,
                        help="Number of hidden channels per block. Default None.")
    parser.add_argument("--conv3d_blocks", type=int, nargs="+", default=[0, 0, 0, 1, 1, 2],
                        help="Type of 3d convolutional blocks per stage. Default [0, 0, 0, 1, 1, 2].")
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
    organ_size = sizes[organ_id]

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

    original_train_size, original_val_size = len(training_entries), len(validation_entries)
    print("Original train/val sizes: {}/{}".format(original_train_size, original_val_size))
    training_entries = train_stage1_results.restrict_patient_ids_to_good_series(training_entries)
    validation_entries = val_stage1_results.restrict_patient_ids_to_good_series(validation_entries)


    training_entries = train_stage1_results.restrict_patient_ids_to_organs(training_entries, organ_id)
    validation_entries = val_stage1_results.restrict_patient_ids_to_organs(validation_entries, organ_id)
    print("Remaining training patients after restriction: {}".format(len(training_entries)))
    print("Remaining validation patients after restriction: {}".format(len(validation_entries)))
    training_entries = np.array(training_entries, dtype=np.int32)
    validation_entries = np.array(validation_entries, dtype=np.int32)

    # initialize gpu and global params
    config.parse_args(args)
    weights_tensor = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32, device=config.device)

    # get model directories
    model_dir, prev_model_dir = manager_models.parse_args(args)

    # obtain model and training parameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    second_momentum = args.second_momentum
    disable_rotpos_augmentation = args.disable_rotpos_augmentation
    disable_elastic_augmentation = args.disable_elastic_augmentation
    batch_size = args.batch_size
    volume_depth = args.volume_depth
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    channel_progression = args.channel_progression
    conv3d_blocks = args.conv3d_blocks
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    num_extra_steps = args.num_extra_steps

    print("Epochs: " + str(epochs))
    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Disable rotation and translation augmentation: " + str(disable_rotpos_augmentation))
    print("Disable elastic augmentation: " + str(disable_elastic_augmentation))
    print("Batch size: " + str(batch_size))

    assert type(hidden_blocks) == list, "Blocks must be a list."
    for k in hidden_blocks:
        assert type(k) == int, "Blocks must be a list of integers."
    assert not (channel_progression is None and hidden_channels is None), "Must specify either hidden channels or channel progression."
    assert channel_progression is None or hidden_channels is None, "Cannot specify both hidden channels and channel progression."
    if channel_progression is None:
        print("------------------------ Using ResNet ------------------------")
        assert hidden_channels % 4 == 0, "Hidden channels must be divisible by 4."
        backbone = model_resnet_old.ResNetBackbone(volume_depth * 2, hidden_channels // bottleneck_factor, use_batch_norm=True,
                                                   pyr_height=4, res_conv_blocks=hidden_blocks,
                                                   bottleneck_expansion=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                                                   use_initial_conv=True)
        model = model_resnet_old.ResNetClassifier(backbone, hidden_channels * (2 ** (len(hidden_blocks) - 1)), out_classes=3)
        using_resnet = True
    else:
        print("------------------------ Using ResNet attention ------------------------")
        assert isinstance(channel_progression, list), "Channel progression must be a list."
        assert len(channel_progression) == len(hidden_blocks), "Channel progression must have same length as hidden blocks."
        assert all([type(k) == int for k in channel_progression]), "Channel progression must be a list of integers."
        assert conv3d_blocks is not None, "Conv3D blocks must be specified for ResNet attention."
        assert isinstance(conv3d_blocks, list), "Conv3D blocks must be a list."
        assert len(conv3d_blocks) == len(hidden_blocks), "Conv3D blocks must have same length as hidden blocks."
        model = model_3d_predictor_resnet.ResNet3DClassifier(
            in_channels=1, out_classes=3,
            channel_progression=channel_progression,
            conv3d_blocks=conv3d_blocks,
            res_conv_blocks=hidden_blocks,
            bottleneck_factor=bottleneck_factor,
            input_depth=volume_depth
        )
        using_resnet = False
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
