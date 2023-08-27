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
import image_sampler
import image_sampler_async
import logging_memory_utils
import manager_folds
import manager_models
import model_resnet
import model_resnet_old
import metrics

train_history, train_metrics, val_history, val_metrics = None, None, None, None
ternary_collapsed_binary_weight = None
ternary_weight = None
bowel_weight = None
extravasation_weight = None
any_injury_weight = None

def optimization_loss(probas: torch.Tensor, ground_truth_labels: torch.Tensor, is_binary: bool, weights: torch.Tensor):
    if is_binary:
        gt_float = ground_truth_labels.to(torch.float32)
        sample_weights = gt_float * weights + 1
        return torch.sum(sample_weights *
                         torch.nn.functional.binary_cross_entropy(probas, gt_float, reduction="none")), torch.sum(sample_weights).item()
    else:
        gt_one_hot = torch.nn.functional.one_hot(ground_truth_labels, num_classes=3).to(torch.float32)
        sample_weights = gt_one_hot * weights
        return torch.sum(sample_weights *
                         torch.nn.functional.nll_loss(torch.log(torch.clamp(probas, min=1e-10)),
                                                      ground_truth_labels, reduction="none")), torch.sum(sample_weights).item()

def target_loss(probas: torch.Tensor, ground_truth_labels: torch.Tensor, is_binary: bool):
    pass

def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                         img_sample_batch_: torch.Tensor, labels_batch_: dict[str, torch.Tensor]):
    optimizer_.zero_grad()
    probas = model_(img_sample_batch_)
    preds = {}
    losses = {}
    loss = 0
    for key in labels_batch_:
        gt = labels_batch_[key]
        if key == "bowel":
            weights = bowel_weight
        elif key == "extravasation":
            weights = extravasation_weight
        else:
            if is_label_binary(key, used_labels):
                weights = ternary_collapsed_binary_weight
            else:
                weights = ternary_weight
        if is_label_binary(key, used_labels):
            opt_loss, weight_sum = optimization_loss(probas[key], gt, is_binary=True, weights=weights)
            losses[key] = {"loss": opt_loss, "weight_sum": weight_sum}
            preds[key] = (probas[key] > 0.5).to(torch.long)
        else:
            opt_loss, weight_sum = optimization_loss(probas[key], gt, is_binary=False, weights=weights)
            losses[key] = {"loss": opt_loss, "weight_sum": weight_sum}
            preds[key] = torch.argmax(probas[key], dim=1)
        loss = loss + losses[key]["loss"]
    loss.backward()
    optimizer.step()

    return preds, {key: {"loss":losses[key]["loss"].item(), "weight_sum":losses[key]["weight_sum"]} for key in losses}

def single_validation_step(model_: torch.nn.Module, img_sample_batch_: torch.Tensor, labels_batch_: dict[str, torch.Tensor]):
    probas = model_(img_sample_batch_)
    preds = {}
    losses = {}
    loss = 0
    for key in labels_batch_:
        gt = labels_batch_[key]
        if key == "bowel":
            weights = bowel_weight
        elif key == "extravasation":
            weights = extravasation_weight
        else:
            if is_label_binary(key, used_labels):
                weights = ternary_collapsed_binary_weight
            else:
                weights = ternary_weight
        if is_label_binary(key, used_labels):
            opt_loss, weight_sum = optimization_loss(probas[key], gt, is_binary=True, weights=weights)
            losses[key] = {"loss": opt_loss, "weight_sum": weight_sum}
            preds[key] = (probas[key] > 0.5).to(torch.long)
        else:
            opt_loss, weight_sum = optimization_loss(probas[key], gt, is_binary=False, weights=weights)
            losses[key] = {"loss": opt_loss, "weight_sum": weight_sum}
            preds[key] = torch.argmax(probas[key], dim=1)
        loss = loss + losses[key]["loss"]

    return preds, {key: {"loss": losses[key]["loss"].item(), "weight_sum": losses[key]["weight_sum"]} for key in losses}

def labels_to_tensor(labels: dict[str, np.ndarray]):
    labels_tensor = {}
    for key in labels:
        labels_tensor[key] = torch.tensor(labels[key], dtype=torch.long, device=config.device)
    return labels_tensor

def training_step(record:bool):
    if record:
        for key in train_metrics:
            train_metrics[key].reset()


    # shuffle
    training_entries_shuffle = np.random.permutation(training_entries)

    # training
    trained = 0
    with tqdm.tqdm(total=len(training_entries)) as pbar:
        while trained < len(training_entries):
            # obtain batch
            current_size = min(len(training_entries) - trained, batch_size)
            patient_ids = training_entries_shuffle[trained:trained + current_size] # patient ids
            series_ids = manager_folds.randomly_pick_series(patient_ids)
            img_data_batch = sampler.obtain_sample_batch(patient_ids, series_ids, slices_random=(not disable_random_slices), augmentation=(not disable_random_augmentation))
            ground_truth_labels_batch = labels_to_tensor(manager_folds.get_patient_status_labels(patient_ids, used_labels))

            preds, individual_losses = single_training_step_compile(model, optimizer, img_data_batch * 255, ground_truth_labels_batch)

            # record
            if record:
                # compute metrics
                with torch.no_grad():
                    for key in train_metrics:
                        if key.endswith("loss"):
                            organ = key[:-5]
                            organ_loss = individual_losses[organ]
                            train_metrics[key].add(organ_loss["loss"], organ_loss["weight_sum"])
                        else:
                            organ = key
                            pred = preds[organ]
                            gt = ground_truth_labels_batch[organ]
                            train_metrics[key].add(pred, gt)


            trained += current_size
            pbar.update(current_size)

    if record:
        current_metrics = {}
        losses = []
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)
            if key.endswith("loss"):
                losses.append(train_metrics[key].get())
        current_metrics["train_loss"] = np.mean(losses)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def validation_step():
    for key in val_metrics:
        val_metrics[key].reset()

    # validating
    validated = 0
    with tqdm.tqdm(total=len(validation_entries)) as pbar:
        with torch.no_grad():
            while validated < len(validation_entries):
                # obtain batch
                current_size = min(len(validation_entries) - validated, batch_size)
                patient_ids = validation_entries[validated:validated + current_size]  # patient ids
                series_ids = manager_folds.randomly_pick_series(patient_ids)
                img_data_batch = sampler.obtain_sample_batch(patient_ids, series_ids, slices_random=False, augmentation=False)
                ground_truth_labels_batch = labels_to_tensor(manager_folds.get_patient_status_labels(patient_ids, used_labels))

                preds, individual_losses = single_validation_step(model, img_data_batch * 255, ground_truth_labels_batch)

                # compute metrics
                with torch.no_grad():
                    for key in val_metrics:
                        if key.endswith("loss"):
                            organ = key[:-5]
                            organ_loss = individual_losses[organ]
                            val_metrics[key].add(organ_loss["loss"], organ_loss["weight_sum"])
                        else:
                            organ = key
                            pred = preds[organ]
                            gt = ground_truth_labels_batch[organ]
                            val_metrics[key].add(pred, gt)

                validated += current_size
                pbar.update(current_size)

    current_metrics = {}
    losses = []
    for key in val_metrics:
        val_metrics[key].write_to_dict(current_metrics)
        if key.endswith("loss"):
            losses.append(val_metrics[key].get())
    current_metrics["val_loss"] = np.mean(losses)

    for key in current_metrics:
        val_history[key].append(current_metrics[key])

def is_label_binary(label: str, used_labels: dict[str, object]):
    return (type(used_labels[label]) == bool) or (used_labels[label] == 1)

def print_history(metrics_history: collections.defaultdict[str, list]):
    for key in metrics_history:
        if ("low" not in key) and ("high" not in key):
            print("{}      {}".format(key, metrics_history[key][-1]))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Train a classifier model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use. Default 1.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate to use. Default 3e-4.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--batch_norm_momentum", type=float, default=0.1, help="Batch normalization momentum to use. Default 0.1.")
    parser.add_argument("--disable_random_slices", action="store_true", help="Whether to disable random slices. Default False.")
    parser.add_argument("--disable_random_augmentation", action="store_true", help="Whether to disable random augmentation. Default False.")
    parser.add_argument("--num_slices", type=int, default=15, help="Number of slices to use. Default 15.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--hidden_channels", type=int, default=16, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 2, 4, 8, 23, 4], help="Number of hidden blocks for ResNet backbone.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false", help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--key_dim", type=int, default=8, help="The key dimension for the attention head. Default 8.")
    parser.add_argument("--proba_head", type=str, default="mean", help="The type of probability head to use. Available options: mean, union. Default mean.")
    parser.add_argument("--use_average_pool_classifier", action="store_true", help="Whether to use average pooling for the classifier. Default False.")
    parser.add_argument("--use_old_resnet", action="store_true", help="Whether to use the old 2D resnet backbone. Default False.")
    parser.add_argument("--num_extra_steps", type=int, default=0, help="Extra steps of gradient descent before the usual step in an epoch. Default 0.")
    parser.add_argument("--async_sampler", action="store_true", help="Whether to use the asynchronous sampler. Default False.")
    parser.add_argument("--bowel", action="store_true", help="Whether to use the bowel labels. Default False.")
    parser.add_argument("--extravasation", action="store_true", help="Whether to use the extravasation labels. Default False.")
    parser.add_argument("--kidney", type=int, default=0, help="Which levels of kidney labels to use. Must be 0, 1, 2. Default 0.")
    parser.add_argument("--liver", type=int, default=0, help="Which levels of liver labels to use. Must be 0, 1, 2. Default 0.")
    parser.add_argument("--spleen", type=int, default=0, help="Which levels of spleen labels to use. Must be 0, 1, 2. Default 0.")
    manager_folds.add_argparse_arguments(parser)
    manager_models.add_argparse_arguments(parser)
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # set device, and initialize weights
    config.parse_args(args)
    ternary_collapsed_binary_weight = torch.tensor(2.0, dtype=torch.float32, device=config.device) # 2 + 1 = 3 = (2 + 4) / 2
    ternary_weight = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32, device=config.device)
    bowel_weight = torch.tensor(1.0, dtype=torch.float32, device=config.device) # 1 + 1 = 2
    extravasation_weight = torch.tensor(5.0, dtype=torch.float32, device=config.device) # 5 + 1 = 6
    any_injury_weight = torch.tensor(5.0, dtype=torch.float32, device=config.device) # 5 + 1 = 6

    # check entries
    training_entries, validation_entries, train_dset_name, val_dset_name = manager_folds.parse_args(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    training_entries = np.array(training_entries)
    validation_entries = np.array(validation_entries)
    print("Training dataset: {}".format(train_dset_name))
    print("Validation dataset: {}".format(val_dset_name))

    # get model directories
    model_dir, prev_model_dir = manager_models.parse_args(args)

    # get which labels to use
    bowel = args.bowel
    extravasation = args.extravasation
    kidney = args.kidney
    liver = args.liver
    spleen = args.spleen
    assert bowel or extravasation or (kidney > 0) or (liver > 0) or (spleen > 0), "Must use at least one label."
    assert kidney in [0, 1, 2], "Kidney must be 0, 1, or 2."
    assert liver in [0, 1, 2], "Liver must be 0, 1, or 2."
    assert spleen in [0, 1, 2], "Spleen must be 0, 1, or 2."
    used_labels = {"bowel": bowel, "extravasation": extravasation, "kidney": kidney, "liver": liver, "spleen": spleen}
    out_classes = {} # output label heads depend on used labels
    if bowel:
        out_classes["bowel"] = 1
    if extravasation:
        out_classes["extravasation"] = 1
    if kidney > 0:
        if kidney == 1:
            out_classes["kidney"] = 1
        else:
            out_classes["kidney"] = 3
    if liver > 0:
        if liver == 1:
            out_classes["liver"] = 1
        else:
            out_classes["liver"] = 3
    if spleen > 0:
        if spleen == 1:
            out_classes["spleen"] = 1
        else:
            out_classes["spleen"] = 3

    # obtain model and training parameters
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    momentum = args.momentum
    second_momentum = args.second_momentum
    batch_norm_momentum = args.batch_norm_momentum
    disable_random_slices = args.disable_random_slices
    disable_random_augmentation = args.disable_random_augmentation
    num_slices = args.num_slices
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    hidden_channels = args.hidden_channels
    hidden_blocks = args.hidden_blocks
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    key_dim = args.key_dim
    proba_head = args.proba_head
    use_average_pool_classifier = args.use_average_pool_classifier
    use_old_resnet = args.use_old_resnet
    num_extra_steps = args.num_extra_steps
    async_sampler = args.async_sampler

    print("Epochs: " + str(epochs))
    print("Batch size: " + str(batch_size))
    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Batch norm momentum: " + str(batch_norm_momentum))
    print("Disable random slices: " + str(disable_random_slices))
    print("Disable random augmentation: " + str(disable_random_augmentation))
    print("Number of slices: " + str(num_slices))

    assert type(hidden_blocks) == list, "Blocks must be a list."
    for k in hidden_blocks:
        assert type(k) == int, "Blocks must be a list of integers."
    assert proba_head in ["mean", "union"], "Probability head must be mean or union."

    print("Hidden channels: " + str(args.hidden_channels))
    print("Hidden blocks: " + str(hidden_blocks))
    print("Bottleneck factor: " + str(bottleneck_factor))
    print("Key dimension: " + str(key_dim))
    print("Squeeze and excitation: " + str(squeeze_excitation))
    print("Probability head: " + str(proba_head))
    print("Use average pool classifier: " + str(use_average_pool_classifier))
    print("Use old resnet: " + str(use_old_resnet))

    # Create model and optimizer
    model_resnet.BATCH_NORM_MOMENTUM = batch_norm_momentum
    if use_old_resnet:
        backbone = model_resnet_old.ResNetBackbone(in_channels=1, hidden_channels=hidden_channels // bottleneck_factor,
                                               use_batch_norm=True, use_res_conv=True, pyr_height=len(hidden_blocks) - 1,
                                               res_conv_blocks=hidden_blocks, bottleneck_expansion=bottleneck_factor,
                                               squeeze_excitation=squeeze_excitation, use_initial_conv=True)
    else:
        backbone = model_resnet.ResNetBackbone(in_channels=1, hidden_channels=hidden_channels,
                                               normalization_type=model_resnet.INSTANCE_NORM, pyr_height=len(hidden_blocks),
                                               res_conv_blocks=hidden_blocks, bottleneck_factor=bottleneck_factor,
                                               squeeze_excitation=squeeze_excitation)
    if use_average_pool_classifier:
        neck = model_resnet.AveragePoolClassifierNeck(channels=hidden_channels * (2 ** (len(hidden_blocks) - 1)), out_classes=out_classes)
    else:
        neck = model_resnet.PatchAttnClassifierNeck(channels=hidden_channels * (2 ** (len(hidden_blocks) - 1)),
                                                    key_dim=key_dim, out_classes=out_classes, normalization_type=model_resnet.INSTANCE_NORM)
    if proba_head == "mean":
        head = model_resnet.MeanProbaReductionHead(channels=hidden_channels * (2 ** (len(hidden_blocks) - 1)),
                                                    out_classes=out_classes)
    else:
        head = model_resnet.UnionProbaReductionHead(channels=hidden_channels * (2 ** (len(hidden_blocks) - 1)),
                                                  out_classes=out_classes)
    if use_old_resnet:
        model = model_resnet_old.FullClassifier(backbone, neck, head)
    else:
        model = model_resnet.FullClassifier(backbone, neck, head)
    model = model.to(config.device)

    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Optimizer: " + optimizer_type)
    print("Batch norm momentum: " + str(batch_norm_momentum))
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
        "model": "Multilabel classifier",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "second_momentum": second_momentum,
        "batch_norm_momentum": batch_norm_momentum,
        "disable_random_slices": disable_random_slices,
        "disable_random_augmentation": disable_random_augmentation,
        "num_slices": num_slices,
        "optimizer": optimizer_type,
        "epochs_per_save": epochs_per_save,
        "hidden_channels": hidden_channels,
        "hidden_blocks": hidden_blocks,
        "bottleneck_factor": bottleneck_factor,
        "squeeze_excitation": squeeze_excitation,
        "key_dim": key_dim,
        "proba_head": proba_head,
        "use_average_pool_classifier": use_average_pool_classifier,
        "num_extra_steps": num_extra_steps,
        "async_sampler": async_sampler,
        "labels": used_labels,
        "train_dataset": train_dset_name,
        "val_dataset": val_dset_name,
        "training_script": "training_classifier.py",
    }

    # Save the model config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    print("Using asynchronous sampler: " + str(async_sampler))
    if async_sampler:
        sampler = image_sampler_async.ImageSamplerAsync(max_batch_size=batch_size, device=config.device, num_slices=num_slices, sampler_name=train_dset_name)
    else:
        sampler = image_sampler.ImageSampler(num_slices=num_slices)

    # Create the metrics
    train_history = collections.defaultdict(list)
    train_metrics = {}
    val_history = collections.defaultdict(list)
    val_metrics = {}
    if bowel:
        train_metrics["bowel"] = metrics.BinaryMetrics(name="train_bowel")
        val_metrics["bowel"] = metrics.BinaryMetrics(name="val_bowel")
        train_metrics["bowel_loss"] = metrics.NumericalMetric(name="train_bowel_loss")
        val_metrics["bowel_loss"] = metrics.NumericalMetric(name="val_bowel_loss")
    if extravasation:
        train_metrics["extravasation"] = metrics.BinaryMetrics(name="train_extravasation")
        val_metrics["extravasation"] = metrics.BinaryMetrics(name="val_extravasation")
        train_metrics["extravasation_loss"] = metrics.NumericalMetric(name="train_extravasation_loss")
        val_metrics["extravasation_loss"] = metrics.NumericalMetric(name="val_extravasation_loss")
    if kidney > 0:
        if kidney == 1:
            train_metrics["kidney"] = metrics.BinaryMetrics(name="train_kidney")
            val_metrics["kidney"] = metrics.BinaryMetrics(name="val_kidney")
        elif kidney == 2:
            train_metrics["kidney"] = metrics.TernaryMetrics(name="train_kidney")
            val_metrics["kidney"] = metrics.TernaryMetrics(name="val_kidney")
        train_metrics["kidney_loss"] = metrics.NumericalMetric(name="train_kidney_loss")
        val_metrics["kidney_loss"] = metrics.NumericalMetric(name="val_kidney_loss")
    if liver > 0:
        if liver == 1:
            train_metrics["liver"] = metrics.BinaryMetrics(name="train_liver")
            val_metrics["liver"] = metrics.BinaryMetrics(name="val_liver")
        elif liver == 2:
            train_metrics["liver"] = metrics.TernaryMetrics(name="train_liver")
            val_metrics["liver"] = metrics.TernaryMetrics(name="val_liver")
        train_metrics["liver_loss"] = metrics.NumericalMetric(name="train_liver_loss")
        val_metrics["liver_loss"] = metrics.NumericalMetric(name="val_liver_loss")
    if spleen > 0:
        if spleen == 1:
            train_metrics["spleen"] = metrics.BinaryMetrics(name="train_spleen")
            val_metrics["spleen"] = metrics.BinaryMetrics(name="val_spleen")
        elif spleen == 2:
            train_metrics["spleen"] = metrics.TernaryMetrics(name="train_spleen")
            val_metrics["spleen"] = metrics.TernaryMetrics(name="val_spleen")
        train_metrics["spleen_loss"] = metrics.NumericalMetric(name="train_spleen_loss")
        val_metrics["spleen_loss"] = metrics.NumericalMetric(name="val_spleen_loss")

    # Compile
    single_training_step_compile = single_training_step#torch.compile(single_training_step)

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
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_{}_substep{}.pt".format(epoch, k)))


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

    sampler.close()
