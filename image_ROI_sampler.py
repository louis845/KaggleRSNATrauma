import h5py
import os
import numpy as np
import torch
import torchvision.transforms.functional
import pandas as pd

import config

shape_info = pd.read_csv("data_hdf5_cropped/shape_info.csv", index_col=1)

def compute_max_angle(height: int, width: int):
    diagonal = np.sqrt(float(height ** 2 + width ** 2))
    if diagonal <= 502:
        max_h_angle = np.pi / 2
        max_w_angle = 0.0
    else:
        max_h_angle = np.arcsin(502 / diagonal)
        max_w_angle = np.arccos(566 / diagonal)

    diag_angle = np.arctan2(height, width)

    return min(max_h_angle - diag_angle, diag_angle - max_w_angle)

def load_image(patient_id: str, series_id: str,
               slices=15,
               slice_region_width = 9,
               slices_random=False,
               augmentation=False) -> (torch.Tensor, torch.Tensor):
    assert slice_region_width % 2 == 1, "slice_region_width must be odd"
    slice_region_radius = (slice_region_width - 1) / 2

    # get slope and slice diff corresponding to 1cm
    slope = shape_info.loc[series_id, "mean_slope"]
    slope_abs = np.abs(slope)
    slice_diff = 10 / slope_abs
    if slice_diff > 1:
        slice_span = np.linspace(-int(slice_region_radius * slice_diff), int(slice_region_radius * slice_diff),
                                 slice_region_width, dtype=np.int32)
        contracted = False
        slice_depth = slice_region_width
    else:
        slice_span = np.arange(-int(slice_region_radius * slice_diff), int(slice_region_radius * slice_diff) + 1, dtype=np.int32)
        contracted = True
        slice_depth = len(slice_span)

    with torch.no_grad():
        with h5py.File(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "ct_3D_image.hdf5"), "r") as f:
            ct_3D_image = f["ct_3D_image"]
            total_slices = ct_3D_image.shape[0]
            original_height, original_width = ct_3D_image.shape[1], ct_3D_image.shape[2]
            max_angle = compute_max_angle(original_height, original_width)
            # at most 15 degrees
            maxdev = min(max_angle, np.pi / 12)
            if maxdev > 0.00872664626:  # if max deviation <= 0.5 degrees, we don't rotate
                angle = np.random.uniform(-maxdev, maxdev)
            else:
                angle = 0.0

        # randomly pick slices, region of interest
        slice_poses = np.linspace(0, total_slices - 1, slices + 2, dtype=np.int32)[1:-1] # equidistant
        if slices_random:
            dist = (np.min(np.diff(slice_poses)) // 2) - 1
            if dist > 1:
                slice_poses = slice_poses + np.random.randint(-dist, dist + 1, size=slices)
                slice_poses = np.clip(np.sort(slice_poses), -slice_span[0], total_slices - 1 - slice_span[-1])

        # sample the images and the segmentation now
        image = torch.zeros((slices, 1, slice_depth, original_height, original_width), dtype=torch.float32, device=config.device)
        segmentations = torch.zeros((slices, 5, slice_depth, original_height, original_width), dtype=torch.float32, device=config.device)
        for k in range(slices):
            slice_pos = slice_poses[k]
            with h5py.File(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "ct_3D_image.hdf5"), "r") as f:
                ct_3D_image = f["ct_3D_image"]
                image_slice = ct_3D_image[slice_pos + slice_span, ...]
            with h5py.File(os.path.join("data_segmentation_hdf_cropped", str(series_id) + ".hdf5"), "r") as f:
                segmentation_3D_image = f["segmentation_arr"]
                segmentation_slice = segmentation_3D_image[slice_pos + slice_span, ...]
            image[k, 0, ...].copy_(torch.from_numpy(image_slice))
            segmentations[k, ...].copy_(torch.from_numpy(segmentation_slice).permute((3, 0, 1, 2)))

        # reshape the depth dimension if contracted
        if contracted:
            image = torchvision.transforms.functional.interpolate(image,
                                size=(slice_region_width, original_height, original_width), mode="nearest")
            segmentations = torchvision.transforms.functional.interpolate(segmentations,
                                size=(slice_region_width, original_height, original_width), mode="nearest")

        # whether augmentation or not, we return a (slices, C, slice_depth, 512, 576) image
        if augmentation:
            # rotate
            image = torchvision.transforms.functional.rotate(image, angle * 180 / np.pi, expand=True, fill=0.0)
            segmentations = torchvision.transforms.functional.rotate(segmentations, angle * 180 / np.pi, expand=True, fill=0.0)

            # expand randomly
            assert image.shape[3] <= 512 and image.shape[4] <= 576
            assert segmentations.shape[3] <= 512 and segmentations.shape[4] <= 576
            height_required = 512 - image.shape[3]
            width_required = 576 - image.shape[4]

            top = np.random.randint(0, height_required + 1)
            bottom = height_required - top
            left = np.random.randint(0, width_required + 1)
            right = width_required - left

            image = torch.nn.functional.pad(image, (left, right, top, bottom))
            segmentations = torch.nn.functional.pad(segmentations, (left, right, top, bottom))
        else:
            top = (512 - image.shape[1]) // 2
            bottom = 512 - image.shape[1] - top
            left = (576 - image.shape[2]) // 2
            right = 576 - image.shape[2] - left

            image = torch.nn.functional.pad(image, (left, right, top, bottom))
            segmentations = torch.nn.functional.pad(segmentations, (left, right, top, bottom))

    return image, segmentations
