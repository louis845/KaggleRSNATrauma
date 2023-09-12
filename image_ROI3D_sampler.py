import h5py
import os
import numpy as np
import torch
import torchvision.transforms.functional
import pandas as pd

import config
import image_sampler_augmentations
import manager_segmentations

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

def load_image(patient_id: str,
               series_id: str,
               segmentation_folder: str, # either manager_segmentations.EXPERT_SEGMENTATION_FOLDER, manager_segmentations.TSM_SEGMENTATION_FOLDER or None, where None means do not load it
               slices = 50,
               translate_rotate_augmentation=False,
               elastic_augmentation=False) -> (torch.Tensor, torch.Tensor):
    # get slope and slice stride corresponding to 0.5cm
    z_positions = np.load(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "z_positions.npy"))
    
    slope_abs = np.abs(slope)
    slice_stride = 5 / slope_abs
    if slice_stride > 1:
        slice_span = np.linspace(-int(slice_region_radius * slice_stride), int(slice_region_radius * slice_stride),
                                 slice_region_depth, dtype=np.int32)
        slice_span[slice_region_radius] = 0
        contracted = False
        loaded_temp_depth = slice_region_depth
    else:
        slice_span = np.arange(-int(slice_region_radius * slice_stride), int(slice_region_radius * slice_stride) + 1, dtype=np.int32)
        contracted = True
        loaded_temp_depth = len(slice_span)

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
        slice_poses = np.linspace(-slice_span[0], total_slices - 1 - slice_span[-1], slices + 2, dtype=np.int32)[1:-1] # equidistant
        if slices_random:
            dist = (np.min(np.diff(slice_poses)) // 2) - 1
            if dist > 1:
                slice_poses = slice_poses + np.random.randint(-dist, dist + 1, size=slices)
                slice_poses = np.sort(slice_poses)
        slice_poses = np.clip(slice_poses, -slice_span[0], total_slices - 1 - slice_span[-1])

        # sample the images and the segmentation now
        image = torch.zeros((slices, 1, loaded_temp_depth, original_height, original_width), dtype=torch.float32, device=config.device)
        if segmentation_region_depth == -1:
            segmentations = None
        elif segmentation_region_depth == 1:
            segmentations = torch.zeros((slices, 4, original_height, original_width), dtype=torch.float32, device=config.device)
        else:
            segmentations = torch.zeros((slices, 4, loaded_temp_depth, original_height, original_width), dtype=torch.float32, device=config.device)
        for k in range(slices):
            slice_pos = slice_poses[k]
            cur_slice_depths = slice_pos + slice_span

            # apply depthwise elastic deformation if necessary
            if elastic_augmentation and not contracted:
                min_slice = np.min(cur_slice_depths)
                max_slice = np.max(cur_slice_depths)
                dist = (np.min(np.diff(cur_slice_depths)) // 4) - 1
                if dist > 1:
                    cur_slice_depths = cur_slice_depths + np.random.randint(-dist, dist + 1, size=loaded_temp_depth)
                    cur_slice_depths = np.clip(cur_slice_depths, min_slice, max_slice)
                    cur_slice_depths[slice_region_radius] = slice_pos # make sure the center is the same

            with h5py.File(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "ct_3D_image.hdf5"), "r") as f:
                ct_3D_image = f["ct_3D_image"]
                image_slice = ct_3D_image[cur_slice_depths, ...]
            image_slice = torch.from_numpy(image_slice)

            if segmentation_region_depth != -1:
                with h5py.File(os.path.join(segmentation_folder, str(series_id) + ".hdf5"), "r") as f:
                    segmentation_3D_image = f["segmentation_arr"]
                    if segmentation_region_depth == 1:
                        segmentation_raw = segmentation_3D_image[slice_pos, ...].astype(dtype=bool)
                        segmentation_slice = np.zeros((original_height, original_width, 4), dtype=bool)
                        segmentation_slice[..., :2] = segmentation_raw[..., :2]
                        segmentation_slice[..., 2] = np.any(segmentation_raw[..., 2:4], axis=-1)
                        segmentation_slice[..., 3] = segmentation_raw[..., 4]
                    else:
                        segmentation_raw = segmentation_3D_image[cur_slice_depths, ...].astype(dtype=bool)
                        segmentation_slice = np.zeros((loaded_temp_depth, original_height, original_width, 4), dtype=bool)
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
            if contracted:
                # 2d elastic deformation, uniform over depth, but different over slices
                displacement_field = torch.tensor(np.concatenate([image_sampler_augmentations.generate_displacement_field(original_width, original_height)
                                        for k in range(slices)], axis=0), dtype=torch.float32, device=config.device)
                image = image_sampler_augmentations.apply_displacement_field3D_simple(image.reshape(slices, loaded_temp_depth, original_height, original_width),
                                                                                 displacement_field).view(slices, 1, loaded_temp_depth, original_height, original_width)
                if segmentation_region_depth != -1:
                    if segmentation_region_depth > 1:
                        segmentations = segmentations.reshape(slices, 4 * loaded_temp_depth, original_height, original_width)
                    segmentations = image_sampler_augmentations.apply_displacement_field3D_simple(segmentations, displacement_field)
                    if segmentation_region_depth > 1:
                        segmentations = segmentations.view(slices, 4, loaded_temp_depth, original_height, original_width)
            else:
                # 3d elastic deformation (varying 2d elastic deformation over depth), and also varying over slices
                displacement_field = torch.stack([image_sampler_augmentations.generate_displacement_field3D(original_width, original_height, loaded_temp_depth, # (slices, loaded_temp_depth, H, W, 2)
                                                                                                kernel_depth_span=[0.3, 0.7, 1, 0.7, 0.3], device=config.device) for k in range(slices)], dim=0)
                assert displacement_field.shape == (slices, loaded_temp_depth, original_height, original_width, 2)
                image = image_sampler_augmentations.apply_displacement_field3D_simple(image.reshape(slices * loaded_temp_depth, 1, original_height, original_width),
                                                                                      displacement_field.view(slices * loaded_temp_depth, original_height, original_width, 2))\
                                .view(slices, 1, loaded_temp_depth, original_height, original_width)
                if segmentation_region_depth != -1:
                    if segmentation_region_depth > 1:
                        segmentations = segmentations.permute(0, 2, 1, 3, 4).reshape(slices * loaded_temp_depth, 4, original_height, original_width)
                        segmentations = image_sampler_augmentations.apply_displacement_field3D_simple(segmentations,
                                                                                        displacement_field.view(slices * loaded_temp_depth, original_height, original_width, 2))
                        segmentations = segmentations.view(slices, loaded_temp_depth, 4, original_height, original_width).permute(0, 2, 1, 3, 4)
                    else: # apply deformation in center slice only
                        segmentations = image_sampler_augmentations.apply_displacement_field3D_simple(segmentations, displacement_field[:, slice_region_radius, ...])


        # flip along the depth dimension if slope > 0
        if slope > 0:
            image = image.flip(2)
            if segmentation_region_depth > 1:
                segmentations = segmentations.flip(2)

        # reshape the depth dimension if contracted
        if contracted:
            image = torch.nn.functional.interpolate(image,
                                size=(slice_region_depth, original_height, original_width), mode="trilinear")
            if segmentation_region_depth > 1:
                segmentations = (torch.nn.functional.interpolate(segmentations,
                                size=(slice_region_depth, original_height, original_width), mode="trilinear") > 0.5).to(torch.float32)
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

        if injury_labels_depth == -1:
            injury_labels = None
        else:
            injury_labels_file = os.path.join(manager_segmentations.SEGMENTATION_LABELS_FOLDER, str(series_id) + ".npy")
            injury_labels = np.load(injury_labels_file)

            if injury_labels_depth > 1:
                injury_labels = injury_labels[np.expand_dims(slice_poses, axis=-1) + np.expand_dims(slice_span, axis=0), :]
                injury_labels_radius = (injury_labels_depth - 1) // 2
                if contracted:
                    assert loaded_temp_depth % 2 == 1 # This is always odd, look at the code above
                    loaded_temp_depth_radius = (loaded_temp_depth - 1) // 2

                    contraction_ratio = float(loaded_temp_depth) / slice_region_depth
                    injury_labels_radius = min(max(int(injury_labels_radius * contraction_ratio), 1), loaded_temp_depth_radius)

                    injury_labels = injury_labels[:, loaded_temp_depth_radius - injury_labels_radius:loaded_temp_depth_radius + injury_labels_radius + 1, :]
                else:
                    injury_labels = injury_labels[:, slice_region_radius - injury_labels_radius:slice_region_radius + injury_labels_radius + 1, :]

                injury_labels = np.concatenate([
                    np.min(injury_labels[:, :, :3], axis=1), # for liver, spleen, kidney, we require all slices in the vicinity to be positive
                    np.max(injury_labels[:, :, 3:], axis=1), # for bowel, extravasation, we require at least one slice in the vicinity to be positive, since its more localised
                ], axis=-1)
            else:
                injury_labels = injury_labels[slice_poses, :] # extract the central slice only

    return image, segmentations, injury_labels