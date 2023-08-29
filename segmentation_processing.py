import nibabel
import h5py
import pydicom
import numpy as np
import torch
import cv2

def to_class_array(dcm: pydicom.dataset.FileDataset, segmentation_arr: np.ndarray) -> np.ndarray:
    """
    Convert segmentation array to class array.
    """
    segmentation_arr = segmentation_arr.astype(np.uint8).transpose(2, 0, 1)
    segmentation_arr = segmentation_arr[::-1, ...]
    segmentation_arr = np.rot90(segmentation_arr, axes=(1, 2), k=1).copy()

    segmentation_arr = torch.nn.functional.one_hot(torch.from_numpy(segmentation_arr).to(torch.long), num_classes=6).numpy()[..., 1:]
    segmentation_arr = segmentation_arr.astype(np.float32)

    scales = np.array(dcm.PixelSpacing)
    shape = segmentation_arr.shape
    new_shape = (int(shape[1] * scales[0]), int(shape[2] * scales[1]))

    result_arr = np.zeros((shape[0], new_shape[0], new_shape[1], 5), dtype=np.uint8)
    for k in range(shape[0]):
        for i in range(5):
            result_arr[k, :, :, i] = (cv2.resize(segmentation_arr[k, :, :, i], (new_shape[1], new_shape[0])) > 0).astype(np.uint8)

    return result_arr
