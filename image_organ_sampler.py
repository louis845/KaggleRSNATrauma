import h5py
import os
import numpy as np
import torch
import torchvision.transforms.functional
import pandas as pd

import config
import image_ROI_sampler
import image_sampler_augmentations
import manager_stage1_results

def load_series_image_and_organloc(patient_id: str, series_id: str, slice_indices,
                      organ_id: int, target_w: int, target_h: int,
                      segmentation_dataset_folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the image of a series from the hdf5 file
    """

    ## Computes the required location of the image to locate the organ
    with h5py.File(os.path.join(segmentation_dataset_folder, series_id + ".hdf5"), "r") as f:
        organ_location = f["organ_location"][organ_id, ...] > 0 # shape (H, W)
    heights = np.any(organ_location, dim=-1)
    widths = np.any(organ_location, dim=-2)
    heights, widths = np.argwhere(heights), np.argwhere(widths)
    # Compute the organ bounds
    minH, maxH = heights.min(), heights.max() + 1
    minW, maxW = widths.min(), widths.max() + 1
    midH, midW = (minH + maxH) / 2.0, (minW + maxW) / 2.0
    # Expand to required bounds
    required_minH = int(midH - target_h / 2.0)
    required_minW = int(midW - target_w / 2.0)
    required_maxH = required_minH + target_h
    required_maxW = required_minW + target_w

    ## Load the image
    collapsed_nearest_indices, repeats = image_ROI_sampler.consecutive_repeats(slice_indices)
    with h5py.File(os.path.join("data_hdf5_cropped", patient_id, series_id, "ct_3D_image.hdf5"), "r") as f:
        ct_3D_image = f["ct_3D_image"] # shape (D, H, W)
        # If the required bounds are outside the image, we pad the image. First compute the pad
        pad_left = max(0, -required_minW)
        pad_right = max(0, required_maxW - ct_3D_image.shape[2])
        pad_top = max(0, -required_minH)
        pad_bottom = max(0, required_maxH - ct_3D_image.shape[1])
        required_minH = max(0, required_minH)
        required_minW = max(0, required_minW)
        required_maxH = min(ct_3D_image.shape[1], required_maxH)
        required_maxW = min(ct_3D_image.shape[2], required_maxW)
        # Load and pad the image
        image = ct_3D_image[collapsed_nearest_indices, required_minH:required_maxH, required_minW:required_maxW]
    image = np.pad(image, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    image = np.repeat(image, repeats, axis=0) # if there is collapsing, repeat it

    ## Do the same to the organ location
    organ_location = organ_location[required_minH:required_maxH, required_minW:required_maxW].astype(np.uint8)
    organ_location = np.pad(organ_location, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    return image, organ_location
    

def load_image(patient_ids: list,
               series_ids: list,
               organ_id: int,
               stage1_information: manager_stage1_results.Stage1ResultsManager,
               organ_sampling_depth = 9,
               translate_rotate_augmentation=False,
               elastic_augmentation=False,
               boundary_augmentation=False) -> (torch.Tensor, torch.Tensor, np.ndarray):


    # get slope and slice stride corresponding to 0.5cm
    stride_mm = 5 # 0.5cm
    slope = image_ROI_sampler.shape_info.loc[int(series_id), "mean_slope"]
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
