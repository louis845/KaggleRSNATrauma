import gc
import os
import time
import argparse
import json
import traceback
import multiprocessing

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
import metrics


def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                         img_sample_batch_: torch.Tensor, labels_batch_: dict[str, torch.Tensor]):
    optimizer_.zero_grad()
    probas = model_(img_sample_batch_)
    losses = {}
    loss = 0
    for key in labels_batch_:
        gt = labels_batch_[key]
        if is_label_binary(key, used_labels):
            losses[key] = torch.nn.functional.binary_cross_entropy(probas[key], gt, reduction="sum")
        else:
            losses[key] = torch.nn.functional.nll_loss(torch.log(probas[key]), gt, reduction="sum")
        loss = loss + losses[key]
    loss.backward()
    optimizer.step()

    return probas, loss.item(), {key: losses[key].item() for key in losses}

def training_step(record:bool):
    if record:
        current_train_metrics = metrics.BinaryMetrics()

    # shuffle
    training_entries_shuffle = np.random.permutation(training_entries)
    if mixup:
        training_entries_shuffle2 = np.random.permutation(training_entries)

    # training
    trained = 0
    with tqdm.tqdm(total=len(training_entries)) as pbar:
        while trained < len(training_entries):
            # obtain batch
            current_size = min(len(training_entries) - trained, batch_size)
            ctime = time.time()
            if mixup:
                batch_entries = training_entries_shuffle[trained:trained + current_size]
                batch_entries2 = training_entries_shuffle2[trained:trained + current_size]
                img_data_original_batch, img_data_ash_batch, ground_truth_data_batch = sampler.obtain_sample_mixup_split_batch(batch_entries, batch_entries2,
                                                                                        mixup_beta=mixup, augmentation=True)
            else:
                batch_entries = training_entries_shuffle[trained:trained + current_size]
                img_data_original_batch, img_data_ash_batch, ground_truth_data_batch = sampler.obtain_sample_split_batch(batch_entries, augmentation=True)

            output, loss = single_training_step_compile(model, optimizer, img_data_original_batch, img_data_ash_batch, ground_truth_data_batch)

            # record
            if record:
                # compute metrics
                with torch.no_grad():
                    predict = torch.argmax(output, dim=1)
                    actual = torch.argmax(ground_truth_data_batch, dim=1)
                    tp, tn, fp, fn = metrics.compute_metrics(predict, actual)
                    current_train_metrics.update(tp, tn, fp, fn, loss, current_size)

            trained += current_size
            pbar.update(current_size)

    if record:
        current_metrics = current_train_metrics.get_metrics()
        train_metrics.update(current_metrics)

def validation_step():
    current_val_metrics = metrics.BinaryMetrics()
    # validation
    with torch.no_grad():
        tested = 0
        with tqdm.tqdm(total=len(validation_entries)) as pbar:
            while tested < len(validation_entries):
                # obtain batch
                current_size = min(len(validation_entries) - tested, val_batch_size)
                batch_entries = validation_entries[tested:tested + current_size]
                img_data_original_batch, img_data_ash_batch, ground_truth_data_batch = sampler.obtain_sample_split_batch(batch_entries, augmentation=False)

                # forward
                output = model(img_data_original_batch, img_data_ash_batch)
                if use_mixed_loss:
                    loss = mixed_loss(output, ground_truth_data_batch)
                else:
                    loss = focal_loss(output, ground_truth_data_batch)

                # compute metrics
                predict = torch.argmax(output, dim=1)
                actual = torch.argmax(ground_truth_data_batch, dim=1)
                tp, tn, fp, fn = metrics.compute_metrics(predict, actual)
                current_val_metrics.update(tp, tn, fp, fn, loss.item(), current_size)

                tested += current_size
                pbar.update(current_size)

    current_metrics = current_val_metrics.get_metrics()
    val_metrics.update(current_metrics)

def is_label_binary(label: str, used_labels: dict[str, object]):
    return (type(used_labels[label]) == bool) or (used_labels[label] == 1)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Train a classifier model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use. Default 1.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate to use. Default 3e-4.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--batch_norm_momentum", type=float, default=0.1, help="Batch normalization momentum to use. Default 0.1.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[4, 8, 23, 4], help="Number of hidden blocks for ResNet backbone.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false", help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--key_dim", type=int, default=8, help="The key dimension for the attention head. Default 8.")
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

    # set device
    config.parse_args(args)

    # check entries
    training_entries, validation_entries = manager_folds.parse_args(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    training_entries = np.array(training_entries)
    validation_entries = np.array(validation_entries)

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
    out_classes = [] # output label heads depend on used labels
    if bowel:
        out_classes.append(1)
    if extravasation:
        out_classes.append(1)
    if kidney > 0:
        if kidney == 1:
            out_classes.append(1)
        else:
            out_classes.append(3)
    if liver > 0:
        if liver == 1:
            out_classes.append(1)
        else:
            out_classes.append(3)
    if spleen > 0:
        if spleen == 1:
            out_classes.append(1)
        else:
            out_classes.append(3)

    # obtain model and training parameters
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    momentum = args.momentum
    second_momentum = args.second_momentum
    batch_norm_momentum = args.batch_norm_momentum
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    hidden_channels = args.hidden_channels
    hidden_blocks = args.hidden_blocks
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    key_dim = args.key_dim
    mixup = args.mixup
    num_extra_steps = args.num_extra_steps
    async_sampler = args.async_sampler

    assert type(hidden_blocks) == list, "Blocks must be a list."
    for k in hidden_blocks:
        assert type(k) == int, "Blocks must be a list of integers."

    print("Hidden channels: " + str(args.hidden_channels))
    print("Hidden blocks: " + str(hidden_blocks))
    print("Bottleneck factor: " + str(bottleneck_factor))
    print("Key dimension: " + str(key_dim))
    print("Squeeze and excitation: " + str(squeeze_excitation))

    # Create model and optimizer
    model_resnet.BATCH_NORM_MOMENTUM = batch_norm_momentum
    backbone = model_resnet.ResNetBackbone(in_channels=1, hidden_channels=hidden_channels,
                                           normalization_type="batchnorm", pyr_height=len(hidden_blocks),
                                           res_conv_blocks=hidden_blocks, bottleneck_factor=bottleneck_factor,
                                           squeeze_excitation=squeeze_excitation)
    neck = model_resnet.PatchAttnClassifierNeck(channels=hidden_channels * (2 ** (len(hidden_blocks) - 1)),
                                                key_dim=key_dim, out_classes=out_classes)
    head = model_resnet.MeanProbaReductionHead(channels=hidden_channels * (2 ** (len(hidden_blocks) - 1)),
                                                out_classes=out_classes)
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
        "optimizer": optimizer_type,
        "epochs_per_save": epochs_per_save,
        "hidden_channels": hidden_channels,
        "hidden_blocks": hidden_blocks,
        "bottleneck_factor": bottleneck_factor,
        "squeeze_excitation": squeeze_excitation,
        "key_dim": key_dim,
        "num_extra_steps": num_extra_steps,
        "async_sampler": async_sampler,
        "labels": used_labels,
        "training_script": "training_classifier.py",
    }

    # Save the model config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    if async_sampler:
        sampler = image_sampler_async.ImageSamplerAsync(max_batch_size=batch_size, device=config.device)

    # Create the metrics
    train_metrics = {}
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
    train_metrics["loss"] = metrics.NumericalMetric(name="train_loss")
    val_metrics["loss"] = metrics.NumericalMetric(name="val_loss")

    # Compile
    single_training_step_compile = torch.compile(single_training_step)

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
            train_metrics.print_latest("TRAIN")
            val_metrics.print_latest("VALID")
            # save metrics
            train_df = train_metrics.to_dataframe()
            val_df = val_metrics.to_dataframe()
            train_df.to_csv(os.path.join(model_dir, "train_metrics.csv"), index=True)
            val_df.to_csv(os.path.join(model_dir, "val_metrics.csv"), index=True)
            assert len(train_df) == len(val_df), "The number of training and validation epochs must be the same."
            # join the two dataframes, add "train_" suffix to train columns and "val_" suffix to val columns
            df = train_df.join(val_df, lsuffix="_train", rsuffix="_val")
            df.to_csv(os.path.join(model_dir, "metrics.csv"), index=True)

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
    train_df = train_metrics.to_dataframe()
    val_df = val_metrics.to_dataframe()
    train_df.to_csv(os.path.join(model_dir, "train_metrics.csv"), index=True)
    val_df.to_csv(os.path.join(model_dir, "val_metrics.csv"), index=True)
    assert len(train_df) == len(val_df), "The number of training and validation epochs must be the same."
    # join the two dataframes, add "train_" suffix to train columns and "val_" suffix to val columns
    df = train_df.join(val_df, lsuffix="_train", rsuffix="_val")
    df.to_csv(os.path.join(model_dir, "metrics.csv"), index=True)

    memory_logger.close()

    sampler.close()