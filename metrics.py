import numpy as np
import torch
import io

import abc

class Metrics(abc.ABC):
    @abc.abstractmethod
    def add(self, *args):
        pass

    @abc.abstractmethod
    def get(self):
        pass

    @abc.abstractmethod
    def write_to_dict(self, x: dict):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class NumericalMetric(Metrics):
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def add(self, value, batch_size: int):
        self.sum += value
        self.count += batch_size

    def get(self):
        return self.sum / self.count

    def write_to_dict(self, x: dict):
        x[self.name] = self.get()

    def reset(self):
        self.sum = 0.0
        self.count = 0

class BinaryMetrics(Metrics):
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def add(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        y_pred and y_true must be tensors of the same shape. They contain binary values (0 or 1).
        """
        assert y_pred.shape == y_true.shape, "y_pred and y_true must have the same shape"
        assert y_pred.dtype == torch.long and y_true.dtype == torch.long, "y_pred and y_true must be long tensors"
        self.tp += torch.sum(y_pred * y_true).cpu().item()
        self.tn += torch.sum((1 - y_pred) * (1 - y_true)).cpu().item()
        self.fp += torch.sum(y_pred * (1 - y_true)).cpu().item()
        self.fn += torch.sum((1 - y_pred) * y_true).cpu().item()

    def add_direct(self, tp, tn, fp, fn):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def get(self):
        """Compute the accuracy, precision, recall."""
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        if self.tp + self.fp == 0:
            precision = 0.0
        else:
            precision = self.tp / (self.tp + self.fp)
        if self.tp + self.fn == 0:
            recall = 0.0
        else:
            recall = self.tp / (self.tp + self.fn)
        return accuracy, precision, recall

    def write_to_dict(self, x: dict):
        accuracy, precision, recall = self.get()
        x[self.name + "_accuracy"] = accuracy
        x[self.name + "_precision"] = precision
        x[self.name + "_recall"] = recall

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def get_iou(self):
        if self.tp + self.fp + self.fn == 0:
            iou = 1.0
        else:
            iou = self.tp / (self.tp + self.fp + self.fn)
        return iou

    def report_print(self, report_iou=False):
        accuracy, precision, recall = self.get()
        print("--------------------- {} ---------------------".format(self.name))
        if report_iou:
            iou = self.get_iou()
            print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, IoU: {:.4f}".format(accuracy, precision, recall, iou))
        else:
            print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(accuracy, precision, recall))

    def report_print_to_file(self, file: io.TextIOWrapper, report_iou=False):
        accuracy, precision, recall = self.get()
        file.write("--------------------- {} ---------------------\n".format(self.name))
        if report_iou:
            iou = self.get_iou()
            file.write("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, IoU: {:.4f}\n".format(accuracy, precision, recall, iou))
        else:
            file.write("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n".format(accuracy, precision, recall))

class TernaryMetrics(Metrics):
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def add(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        y_pred and y_true must be 1D tensors of the same shape. They contain ternary values (0 to 2).
        """
        assert y_pred.shape == y_true.shape, "y_pred and y_true must have the same shape"
        assert y_pred.dtype == torch.long and y_true.dtype == torch.long, "y_pred and y_true must be long tensors"
        self.tp += torch.sum((y_pred > 0) & (y_true > 0)).cpu().item()
        self.tn += torch.sum((y_pred == 0) & (y_true == 0)).cpu().item()
        self.fp += torch.sum((y_pred > 0) & (y_true == 0)).cpu().item()
        self.fn += torch.sum((y_pred == 0) & (y_true > 0)).cpu().item()

        self.low_tp += torch.sum((y_pred == 1) & (y_true == 1)).cpu().item()
        self.low_tn += torch.sum((y_pred != 1) & (y_true != 1)).cpu().item()
        self.low_fp += torch.sum((y_pred == 1) & (y_true != 1)).cpu().item()
        self.low_fn += torch.sum((y_pred != 1) & (y_true == 1)).cpu().item()

        self.high_tp += torch.sum((y_pred == 2) & (y_true == 2)).cpu().item()
        self.high_tn += torch.sum((y_pred != 2) & (y_true != 2)).cpu().item()
        self.high_fp += torch.sum((y_pred == 2) & (y_true != 2)).cpu().item()
        self.high_fn += torch.sum((y_pred != 2) & (y_true == 2)).cpu().item()

        self.correct += torch.sum(y_pred == y_true).cpu().item()

    def get(self):
        """Compute the accuracy, precision, recall."""
        overall_accuracy = self.correct / (self.tp + self.tn + self.fp + self.fn)

        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        if self.tp + self.fp == 0:
            precision = 0.0
        else:
            precision = self.tp / (self.tp + self.fp)
        if self.tp + self.fn == 0:
            recall = 0.0
        else:
            recall = self.tp / (self.tp + self.fn)

        low_accuracy = (self.low_tp + self.low_tn) / (self.low_tp + self.low_tn + self.low_fp + self.low_fn)
        if self.low_tp + self.low_fp == 0:
            low_precision = 0.0
        else:
            low_precision = self.low_tp / (self.low_tp + self.low_fp)
        if self.low_tp + self.low_fn == 0:
            low_recall = 0.0
        else:
            low_recall = self.low_tp / (self.low_tp + self.low_fn)

        high_accuracy = (self.high_tp + self.high_tn) / (self.high_tp + self.high_tn + self.high_fp + self.high_fn)
        if self.high_tp + self.high_fp == 0:
            high_precision = 0.0
        else:
            high_precision = self.high_tp / (self.high_tp + self.high_fp)
        if self.high_tp + self.high_fn == 0:
            high_recall = 0.0
        else:
            high_recall = self.high_tp / (self.high_tp + self.high_fn)

        return overall_accuracy, accuracy, precision, recall, low_accuracy,\
            low_precision, low_recall, high_accuracy, high_precision, high_recall

    def write_to_dict(self, x: dict):
        overall_accuracy, accuracy, precision, recall, low_accuracy,\
            low_precision, low_recall, high_accuracy, high_precision, high_recall = self.get()
        x[self.name + "_overall_accuracy"] = overall_accuracy
        x[self.name + "_accuracy"] = accuracy
        x[self.name + "_precision"] = precision
        x[self.name + "_recall"] = recall
        x[self.name + "_low_accuracy"] = low_accuracy
        x[self.name + "_low_precision"] = low_precision
        x[self.name + "_low_recall"] = low_recall
        x[self.name + "_high_accuracy"] = high_accuracy
        x[self.name + "_high_precision"] = high_precision
        x[self.name + "_high_recall"] = high_recall

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.low_tp = 0
        self.low_tn = 0
        self.low_fp = 0
        self.low_fn = 0

        self.high_tp = 0
        self.high_tn = 0
        self.high_fp = 0
        self.high_fn = 0

        self.correct = 0
