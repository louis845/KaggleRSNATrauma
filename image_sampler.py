import h5py
import os
import numpy as np
import torch

def load_image(patient_id: str, series_id: str, slices=15, slices_random=False, augmentation=False) -> np.ndarray:
    with h5py.File(os.path.join("data_hdf5", patient_id, series_id, "ct_3D_image.hdf5"), "r") as f:
        ct_3D_image = f["ct_3D_image"]
        total_slices = ct_3D_image.shape[0]

        # equidistant
        slices_pos = np.linspace(0, total_slices - 1, slices + 2, dtype=np.int32)[1:-1]

        # randomly pick slices
        if slices_random:
            dist = np.min(np.diff(slices_pos)) // 2
            slices_pos = slices_pos + np.random.randint(-dist, dist + 1, size=slices)
            slices_pos = np.clip(np.sort(slices_pos), 0, total_slices - 1)

        image = ct_3D_image[slices_pos, :, :]
        image = torch.from_numpy(image).to(torch.float32)

        # whether augmentation or not, we return a (slices, 640, 640) image
        if augmentation:
            h' = w sin theta + h cos theta
            w' = w cos theta + h sin theta
        else:
            top = (640 - image.shape[1]) // 2
            bottom = 640 - image.shape[1] - top
            left = (640 - image.shape[2]) // 2
            right = 640 - image.shape[2] - left

            image = torch.nn.functional.pad(image, (left, right, top, bottom))


    return image