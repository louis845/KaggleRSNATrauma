import h5py
import os
import numpy as np
import torch
import torchvision.transforms.functional
import pandas as pd

import config
import image_sampler_augmentations
import manager_segmentations

shape_info = pd.read_csv("data_hdf5_cropped/shape_info.csv", index_col=1)

def compute_max_angle(height: int, width: int):
    diagonal = np.sqrt(float(height ** 2 + width ** 2))
    if diagonal <= 502:
        max_h_angle = np.pi / 2
    else:
        max_h_angle = np.arcsin(502 / diagonal)
    if diagonal <= 566:
        max_w_angle = 0.0
    else:
        max_w_angle = np.arccos(566 / diagonal)

    diag_angle = np.arctan2(height, width)

    return min(max_h_angle - diag_angle, diag_angle - max_w_angle)

def find_closest(Z: np.ndarray, x: np.ndarray):
    """Given Z and x, find the indices i(j) such that Z[i(j)] is the closest element to x[j]"""
    # find the indices i such that Z[i] is the first element greater than or equal to x
    i = np.searchsorted(Z, x)
    # Handle cases where x[j] is greater than all elements in Z
    i[i == len(Z)] = len(Z) - 1
    # For all locations (i > 0), if x[j] is closer to Z[i - 1] than Z[i], decrement i
    # Note that when i = 0, then Z[i - 1] would be the last element, which doesn't make sense.
    # But it will be masked out by the mask anyway.
    mask = (i > 0) & ((np.abs(Z[i - 1] - x) <= np.abs(Z[i] - x)))
    i[mask] -= 1
    return i

def consecutive_repeats(arr):
    if len(arr) == 0:
        return np.array([])
    else:
        diff = np.diff(arr)
        idx = np.argwhere(diff != 0).squeeze(-1) + 1
        idx = np.concatenate([np.array([0]), idx, np.array([len(arr)])], axis=0)
        repeats = np.diff(idx)
        return arr[idx[:-1]], repeats

def get_nearest_slice_indices(slice_idx, z_positions: np.ndarray, stride_mm, depth, is_flipped):
    """
    Same as the implementation in stage1_organ_segmentation.py
    """
    assert depth % 2 == 1, "depth must be an odd number"
    depth_radius = (depth - 1) // 2

    current_zpos = z_positions[slice_idx]
    expected_zposes = np.arange(current_zpos - depth_radius * stride_mm,
                                current_zpos + (depth_radius + 1) * stride_mm, stride_mm)

    # find nearest slice indices for the given z positions
    if is_flipped:
        nearest_slice_indices = find_closest(z_positions, expected_zposes)
    else:
        nearest_slice_indices = (z_positions.shape[0] - 1 -
                                 find_closest(z_positions[::-1], expected_zposes))[::-1]
    nearest_slice_indices[depth_radius] = slice_idx
    return nearest_slice_indices

def load_slice_from_hdf(hdf_file: h5py.File, slice_indices, array_str: str):
    collapsed_nearest_indices, repeats = consecutive_repeats(slice_indices)

    local_slice_image = hdf_file[array_str][collapsed_nearest_indices, ...]
    local_slice_image = np.repeat(local_slice_image, repeats, axis=0)

    return local_slice_image

def load_image(patient_id: str,
               series_id: str,
               segmentation_folder: str, # either manager_segmentations.EXPERT_SEGMENTATION_FOLDER, manager_segmentations.TSM_SEGMENTATION_FOLDER or None, where None means do not load it
               slices = 15,
               slice_region_depth = 9,
               segmentation_region_depth = 1,
               slices_random=False,
               translate_rotate_augmentation=False,
               elastic_augmentation=False,
               boundary_augmentation=False) -> (torch.Tensor, torch.Tensor, np.ndarray):
    if segmentation_folder is None:
        segmentation_region_depth = -1
    assert slice_region_depth % 2 == 1, "slice_region_depth must be odd"
    assert segmentation_region_depth % 2 == 1, "segmentation_region_depth must be odd"
    assert segmentation_region_depth <= slice_region_depth, "segmentation_region_depth must be less than or equal to slice_region_depth"
    slice_region_radius = (slice_region_depth - 1) // 2

    # get slope and slice stride corresponding to 0.5cm
    stride_mm = 5 # 0.5cm
    slope = shape_info.loc[int(series_id), "mean_slope"]
    is_flipped = slope > 0.0
    z_positions = np.load(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "z_positions.npy"))

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
        interest_min, interest_max = slice_poses[0], slice_poses[-1]
        if slices_random:
            dist = (np.min(np.diff(slice_poses)) // 2) - 1
            if dist > 1:
                slice_poses = slice_poses + np.random.randint(-dist, dist + 1, size=slices)
                slice_poses = np.sort(slice_poses)
        slice_poses = np.clip(slice_poses, interest_min, interest_max)

        # sample the images and the segmentation now
        image = torch.zeros((slices, 1, slice_region_depth, original_height, original_width), dtype=torch.float32, device=config.device)
        if segmentation_region_depth == -1:
            segmentations = None
        elif segmentation_region_depth == 1:
            segmentations = torch.zeros((slices, 4, original_height, original_width), dtype=torch.float32, device=config.device)
        else:
            segmentations = torch.zeros((slices, 4, slice_region_depth, original_height, original_width), dtype=torch.float32, device=config.device)
        for k in range(slices):
            slice_pos = slice_poses[k]
            cur_slice_depths = get_nearest_slice_indices(slice_pos, z_positions, stride_mm, slice_region_depth, is_flipped)

            # apply depthwise elastic deformation if necessary
            if elastic_augmentation:
                min_slice = np.min(cur_slice_depths)
                max_slice = np.max(cur_slice_depths)
                dist = (np.min(np.diff(cur_slice_depths)) // 4) - 1
                if dist > 1:
                    cur_slice_depths = cur_slice_depths + np.random.randint(-dist, dist + 1, size=slice_region_depth)
                    cur_slice_depths = np.clip(cur_slice_depths, min_slice, max_slice)
                    cur_slice_depths[slice_region_radius] = slice_pos # make sure the center is the same

            # apply boundary augmentation if necessary
            if boundary_augmentation:
                boundary_augment = np.random.randint(0, 3)
                if boundary_augment == 1:
                    random_boundary = np.random.randint(0, slice_region_radius)
                    cur_slice_depths[:(random_boundary + 1)] = cur_slice_depths[random_boundary + 1]
                elif boundary_augment == 2:
                    random_boundary = np.random.randint(0, slice_region_radius)
                    cur_slice_depths[-(random_boundary + 1):] = cur_slice_depths[-(random_boundary + 2)]

            cur_slice_depths = np.clip(cur_slice_depths, 0, total_slices - 1)
            with h5py.File(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "ct_3D_image.hdf5"), "r") as f:
                image_slice = load_slice_from_hdf(f, cur_slice_depths, "ct_3D_image")
            image_slice = torch.from_numpy(image_slice)

            if segmentation_region_depth != -1:
                with h5py.File(os.path.join(segmentation_folder, str(series_id) + ".hdf5"), "r") as f:
                    if segmentation_region_depth == 1:
                        segmentation_3D_image = f["segmentation_arr"]
                        segmentation_raw = segmentation_3D_image[slice_pos, ...].astype(dtype=bool)
                        segmentation_slice = np.zeros((original_height, original_width, 4), dtype=bool)
                        segmentation_slice[..., :2] = segmentation_raw[..., :2]
                        segmentation_slice[..., 2] = np.any(segmentation_raw[..., 2:4], axis=-1)
                        segmentation_slice[..., 3] = segmentation_raw[..., 4]
                    else:
                        segmentation_raw = load_slice_from_hdf(f, cur_slice_depths, "segmentation_arr").astype(dtype=bool)
                        segmentation_slice = np.zeros((slice_region_depth, original_height, original_width, 4), dtype=bool)
                        segmentation_slice[..., :2] = segmentation_raw[..., :2]
                        segmentation_slice[..., 2] = np.any(segmentation_raw[..., 2:4], axis=-1)
                        segmentation_slice[..., 3] = segmentation_raw[..., 4]
                    del segmentation_raw
                if segmentation_region_depth == 1:
                    segmentation_slice = torch.tensor(segmentation_slice, dtype=torch.float32).permute((2, 0, 1))
                else:
                    segmentation_slice = torch.tensor(segmentation_slice, dtype=torch.float32).permute((3, 0, 1, 2))


            image[k, 0, ...].copy_(image_slice, non_blocking=True)
            if segmentation_region_depth != -1:
                segmentations[k, ...].copy_(segmentation_slice, non_blocking=True)

        # apply elastic deformation to height width
        if elastic_augmentation:
            # 3d elastic deformation (varying 2d elastic deformation over depth), and also varying over slices
            displacement_field = torch.stack([image_sampler_augmentations.generate_displacement_field3D(original_width, original_height, slice_region_depth, # (slices, loaded_temp_depth, H, W, 2)
                                                                                            kernel_depth_span=[0.3, 0.7, 1, 0.7, 0.3], device=config.device) for k in range(slices)], dim=0)
            assert displacement_field.shape == (slices, slice_region_depth, original_height, original_width, 2)
            image = image_sampler_augmentations.apply_displacement_field3D_simple(image.reshape(slices * slice_region_depth, 1, original_height, original_width),
                                                                                  displacement_field.view(slices * slice_region_depth, original_height, original_width, 2))\
                            .view(slices, 1, slice_region_depth, original_height, original_width)
            if segmentation_region_depth != -1:
                if segmentation_region_depth > 1:
                    segmentations = segmentations.permute(0, 2, 1, 3, 4).reshape(slices * slice_region_depth, 4, original_height, original_width)
                    segmentations = image_sampler_augmentations.apply_displacement_field3D_simple(segmentations,
                                                                                    displacement_field.view(slices * slice_region_depth, original_height, original_width, 2))
                    segmentations = segmentations.view(slices, slice_region_depth, 4, original_height, original_width).permute(0, 2, 1, 3, 4)
                else: # apply deformation in center slice only
                    segmentations = image_sampler_augmentations.apply_displacement_field3D_simple(segmentations, displacement_field[:, slice_region_radius, ...])


        # flip along the depth dimension if slope > 0
        if slope > 0:
            image = image.flip(2)
            if segmentation_region_depth > 1:
                segmentations = segmentations.flip(2)

        if segmentation_region_depth > 1:
            segmentations = segmentations[:, :, slice_region_radius - (segmentation_region_depth - 1) // 2:slice_region_radius + (segmentation_region_depth + 1) // 2, ...]

        if segmentation_region_depth != -1:
            assert image.shape[-2] == segmentations.shape[-2] and image.shape[-1] == segmentations.shape[-1]
        # whether augmentation or not, we return a (slices, C, slice_depth, 512, 576) image
        if translate_rotate_augmentation:
            # rotate
            image = torchvision.transforms.functional.rotate(image.squeeze(1), angle * 180 / np.pi, expand=True,
                                                             interpolation=torchvision.transforms.InterpolationMode.NEAREST).unsqueeze(1)
            if segmentation_region_depth != -1:
                if segmentation_region_depth == 1:
                    segmentations = torchvision.transforms.functional.rotate(segmentations, angle * 180 / np.pi, expand=True,
                                                                             interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                else:
                    N, C, D, H, W = segmentations.shape
                    segmentations = torchvision.transforms.functional.rotate(segmentations.reshape(N * C * D, 1, H, W), angle * 180 / np.pi, expand=True,
                                                        interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                    N_, _, H, W = segmentations.shape
                    segmentations = segmentations.view(N, C, D, H, W)

            # expand randomly
            assert image.shape[-2] <= 512 and image.shape[-1] <= 576
            if segmentation_region_depth != -1:
                assert segmentations.shape[-2] <= 512 and segmentations.shape[-1] <= 576
                assert image.shape[-2] == segmentations.shape[-2] and image.shape[-1] == segmentations.shape[-1]
            height_required = 512 - image.shape[-2]
            width_required = 576 - image.shape[-1]

            top = np.random.randint(0, height_required + 1)
            bottom = height_required - top
            left = np.random.randint(0, width_required + 1)
            right = width_required - left

            image = torch.nn.functional.pad(image, (left, right, top, bottom))
            if segmentation_region_depth != -1:
                segmentations = torch.nn.functional.pad(segmentations, (left, right, top, bottom))
        else:
            top = (512 - image.shape[-2]) // 2
            bottom = 512 - image.shape[-2] - top
            left = (576 - image.shape[-1]) // 2
            right = 576 - image.shape[-1] - left

            image = torch.nn.functional.pad(image, (left, right, top, bottom))
            if segmentation_region_depth != -1:
                segmentations = torch.nn.functional.pad(segmentations, (left, right, top, bottom))

        if segmentation_region_depth != -1:
            # downscale segmentations by 32 with max pooling
            if segmentation_region_depth == 1:
                segmentations = torch.nn.functional.max_pool2d(segmentations, kernel_size=32, stride=32)
            else:
                segmentations = torch.nn.functional.max_pool2d(segmentations.view(slices, 4 * segmentation_region_depth, 512, 576),
                                                    kernel_size=32, stride=32).view(slices, 4, segmentation_region_depth, 16, 18)

    return image, segmentations
