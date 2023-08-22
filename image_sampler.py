import h5py
import os
import numpy as np
import torch
import torchvision.transforms.functional

import config

def compute_max_angle(height: int, width: int):
    diagonal = np.sqrt(float(height ** 2 + width ** 2))
    if diagonal <= 640:
        max_h_angle = np.pi / 2
        max_w_angle = 0.0
    else:
        max_h_angle = np.arcsin(640 / diagonal)
        max_w_angle = np.arccos(640 / diagonal)

    diag_angle = np.arctan2(height, width)

    return min(max_h_angle - diag_angle, diag_angle - max_w_angle)

def randomly_augment_image(image: torch.Tensor):
    max_angle = compute_max_angle(image.shape[1], image.shape[2])
    # at most 15 degrees
    maxdev = min(max_angle, np.pi / 12)
    if maxdev > 0.00872664626:  # if max deviation <= 0.5 degrees, we don't rotate
        angle = np.random.uniform(-maxdev, maxdev)
    else:
        angle = 0.0

    # rotate
    image = torchvision.transforms.functional.rotate(image, angle * 180 / np.pi, expand=True, fill=0.0)

    # expand randomly
    assert image.shape[1] <= 640 and image.shape[2] <= 640
    height_required = 640 - image.shape[1]
    width_required = 640 - image.shape[2]

    top = np.random.randint(0, height_required + 1)
    bottom = height_required - top
    left = np.random.randint(0, width_required + 1)
    right = width_required - left

    image = torch.nn.functional.pad(image, (left, right, top, bottom))

    return image

def load_image(patient_id: str, series_id: str, slices=15, slices_random=False, augmentation=False) -> np.ndarray:
    with torch.no_grad():
        with h5py.File(os.path.join("data_hdf5", str(patient_id), str(series_id), "ct_3D_image.hdf5"), "r") as f:
            ct_3D_image = f["ct_3D_image"]
            total_slices = ct_3D_image.shape[0]

            # equidistant
            slices_pos = np.linspace(0, total_slices - 1, slices + 2, dtype=np.int32)[1:-1]

            # randomly pick slices
            if slices_random:
                dist = (np.min(np.diff(slices_pos)) // 2) - 1
                if dist > 1:
                    slices_pos = slices_pos + np.random.randint(-dist, dist + 1, size=slices)
                    slices_pos = np.clip(np.sort(slices_pos), 0, total_slices - 1)
            image = ct_3D_image[slices_pos, :, :]
            image = torch.from_numpy(image).to(torch.float32)

        # whether augmentation or not, we return a (slices, 640, 640) image
        if augmentation:
            image = randomly_augment_image(image)
        else:
            top = (640 - image.shape[1]) // 2
            bottom = 640 - image.shape[1] - top
            left = (640 - image.shape[2]) // 2
            right = 640 - image.shape[2] - left

            image = torch.nn.functional.pad(image, (left, right, top, bottom))

    return image.numpy()

def obtain_sample_batch(patient_ids: list[str], series_ids: list[str], slices_random: bool, augmentation: bool):
    assert len(patient_ids) == len(series_ids)
    batch_size = len(patient_ids)

    img_data_batch = torch.zeros((batch_size, 1, 15, 640, 640), dtype=torch.float32, device=config.device)

    for i in range(batch_size):
        img_data = load_image(patient_ids[i], series_ids[i], slices=15,
                              slices_random=slices_random, augmentation=augmentation)
        img_data_batch[i, 0, ...].copy_(torch.from_numpy(img_data), non_blocking=True)

    return img_data_batch

# dummy class
class ImageSampler:
    def obtain_sample_batch(self, patient_ids: list[str], series_ids: list[str], slices_random: bool, augmentation: bool):
        return obtain_sample_batch(patient_ids, series_ids, slices_random, augmentation)

    def close(self):
        pass