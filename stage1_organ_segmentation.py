import torch
import h5py
import numpy as np
import pandas as pd

import model_3d_patch_resnet

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

class OrganSegmentator():

    ct_3d_volume: h5py.File
    z_positions: np.ndarray

    def __init__(self, channel_progression: list[int]=[2, 3, 6, 9, 15, 32, 128, 256, 512, 1024], res_conv3d_blocks=[1, 2, 1, 0, 0, 0],
                 res_conv_blocks=[1, 2, 6, 8, 23, 8], bottleneck_factor=4, squeeze_excitation=True, device="cuda"):
        backbone = model_3d_patch_resnet.ResNet3DBackbone(in_channels=1,
                                                      channel_progression=channel_progression,
                                                      res_conv3d_blocks=res_conv3d_blocks,
                                                      res_conv_blocks=res_conv_blocks,
                                                      bottleneck_factor=bottleneck_factor,
                                                      squeeze_excitation=squeeze_excitation,
                                                      return_3d_features=False)
        # model
        self.model = model_3d_patch_resnet.LocalizedROINet(backbone=self.backbone, num_channels=channel_progression[-1])
        self.model.to(device)
        self.device = device
        self.model_loaded = False

        # loaded data
        self.ct_3d_volume = None
        self.z_positions = None
        self.is_flipped = False
        self.data_loaded = False

    def load_checkpoint(self, model_checkpoint_path: str):
        # for example, model_checkpoint_path = "/path/to/model.pt"
        self.model.load_state_dict(torch.load(model_checkpoint_path))
        self.model.eval()
        self.model_loaded = True

    def load_data(self, ct_3d_volume_path: str, z_positions_path: str):
        if self.data_loaded:
            self.close()
        self.ct_3d_volume = h5py.File(ct_3d_volume_path, "r")
        self.z_positions = np.load(z_positions_path)
        if np.abs(np.diff(self.z_positions)).mean() > 0.0:
            self.is_flipped = True
        self.data_loaded = True

    def close(self):
        if self.data_loaded:
            self.ct_3d_volume.close()
            self.ct_3d_volume = None
            self.z_positions = None
            self.data_loaded = False

    def find_min_max_slice_idxs(self, stride_mm: int=5, depth: int=9):
        assert self.data_loaded, "Please load the data first."
        assert depth % 2 == 1, "depth must be an odd number"
        depth_radius = (depth - 1) // 2

        len_required = depth_radius * stride_mm
        if self.is_flipped:
            z_pos_view = self.z_positions
        else:
            z_pos_view = self.z_positions[::-1]
        min_slice_idx = np.searchsorted(self.z_positions, self.z_positions[0] + len_required, side="left")
        max_slice_idx = np.searchsorted(self.z_positions, self.z_positions[-1] - len_required, side="right") - 1
        if not self.is_flipped:
            min_slice_idx = self.z_positions.shape[0] - 1 - min_slice_idx
            max_slice_idx = self.z_positions.shape[0] - 1 - max_slice_idx
        return min_slice_idx, max_slice_idx

    def load_local_features(self, slice_idx, stride_mm: int=5, depth: int=9):
        assert self.data_loaded, "Please load the data first."
        assert depth % 2 == 1, "depth must be an odd number"
        depth_radius = (depth - 1) // 2

        current_zpos = self.z_positions[slice_idx]
        expected_zposes = np.arange(current_zpos - depth_radius * stride_mm, current_zpos + (depth_radius + 1) * stride_mm, stride_mm)

        # find nearest slice indices for the given z positions
        if self.is_flipped:
            nearest_slice_indices = find_closest(self.z_positions, expected_zposes)
        else:
            nearest_slice_indices = self.z_positions.shape[0] - 1 -\
                                        find_closest(self.z_positions[::-1], expected_zposes)
        nearest_slice_indices[depth_radius] = slice_idx
        local_slice_image = self.ct_3d_volume["ct_3d_volume"][nearest_slice_indices, ...]
        if self.is_flipped:
            local_slice_image = local_slice_image[::-1, ...]
        return local_slice_image

    def predict_at_locations(self, slice_idxs, stride_mm: int=5, depth: int=9):
        assert self.model_loaded, "Please load the model checkpoint first."
        assert self.data_loaded, "Please load the data first."
        assert depth % 2 == 1, "depth must be an odd number"

        image_tensor = torch.stack([
                    torch.tensor(self.load_local_features(slice_idxs[k], stride_mm, depth), dtype=torch.float32,
                             device=self.device) for k in range(len(slice_idxs))], dim=0).unsqueeze(1)
        # the shape should be (batch_size, 1, D, H, W)
        assert image_tensor.shape[:3] == (len(slice_idxs), 1, depth)
        with torch.no_grad():
            # if height > 512, we crop the height at the center to make it 512.
            # conversely, if height < 512, we pad the height at the top and bottom to make it 512.
            if image_tensor.shape[3] > 512:
                image_tensor = image_tensor[..., (image_tensor.shape[3] - 512) // 2:(image_tensor.shape[3] + 512) // 2, :]
            elif image_tensor.shape[3] < 512:
                pad_top = (512 - image_tensor.shape[3]) // 2
                pad_bottom = 512 - image_tensor.shape[3] - pad_top
                image_tensor = torch.nn.functional.pad(image_tensor, (0, 0, pad_top, pad_bottom))
            # if width > 576, we crop the width at the center to make it 576.
            # conversely, if width < 576, we pad the width at the left and right to make it 576.
            if image_tensor.shape[4] > 576:
                image_tensor = image_tensor[..., (image_tensor.shape[4] - 576) // 2:(image_tensor.shape[4] + 576) // 2]
            elif image_tensor.shape[4] < 576:
                pad_left = (576 - image_tensor.shape[4]) // 2
                pad_right = 576 - image_tensor.shape[4] - pad_left
                image_tensor = torch.nn.functional.pad(image_tensor, (pad_left, pad_right, 0, 0))

            preds = self.model(image_tensor) > 0

        # the shape should be (batch_size, 4, H, W)
        assert preds.shape[:2] == (len(slice_idxs), 4)
        return preds


    def predict_organs(self, ct_3d_volume_path: str, z_positions_path: str) -> list[tuple[int, int]]:
        """
        Predicts where the organs are located at. For each organ (liver: 0, spleen: 1, kidney: 2, bowel: 3), we predict
        a 2 tuple, representing the index of the slice number where the organ starts and ends.
        ct_3d_volume_path: path to the 3d volume of the CT scan. For example, /path/to/ct_3d_volume.hdf5
        z_positions_path: path to the z positions of the CT scan. For example, /path/to/z_positions.npy
        """
        assert self.model_loaded, "Please load the model checkpoint first."
        self.load_data(ct_3d_volume_path, z_positions_path)
        min_slice_idx, max_slice_idx = self.find_min_max_slice_idxs()

