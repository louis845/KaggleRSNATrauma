import torch
import h5py
import numpy as np
import os
import argparse

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
        self.model = model_3d_patch_resnet.LocalizedROINet(backbone=backbone, num_channels=channel_progression[-1])
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
        self.is_flipped = np.diff(self.z_positions).mean() > 0.0
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
        min_slice_idx = np.searchsorted(z_pos_view, z_pos_view[0] + len_required, side="left")
        max_slice_idx = np.searchsorted(z_pos_view, z_pos_view[-1] - len_required, side="right") - 1
        if not self.is_flipped:
            min_slice_idx = self.z_positions.shape[0] - 1 - min_slice_idx
            max_slice_idx = self.z_positions.shape[0] - 1 - max_slice_idx
            max_slice_idx, min_slice_idx = min_slice_idx, max_slice_idx
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
        local_slice_image = self.ct_3d_volume["ct_3D_image"][nearest_slice_indices, ...]
        if self.is_flipped:
            local_slice_image = local_slice_image[::-1, ...]
        return local_slice_image

    def predict_at_locations(self, slice_idxs: np.ndarray, stride_mm: int=5, depth: int=9):
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

    @staticmethod
    def reduce_slice(preds):
        with torch.no_grad():
            return torch.any(torch.any(preds, -1), preds, -2).cpu().numpy()


    def predict_organs(self, ct_3d_volume_path: str, z_positions_path: str, batch_size=8,
                            locate_organ_min_gap_ratio=0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts where the organs are located at. For each organ (liver: 0, spleen: 1, kidney: 2, bowel: 3), we predict
        a 2 tuple, representing the index of the slice number where the organ starts and ends.
        ct_3d_volume_path: path to the 3d volume of the CT scan. For example, /path/to/ct_3d_volume.hdf5
        z_positions_path: path to the z positions of the CT scan. For example, /path/to/z_positions.npy
        batch_size: batch size for the prediction
        locate_organ_min_gap_ratio: the minimum gap for the search algorithm. the algorithm terminates when the all gap between
                        predicted slices are smaller than this value. The min gap is this value as the portion of total slices
        
        Returns:
            found_organs: np.ndarray of shape (4,), type bool, representing whether the organ is found or not
            organ_left: np.ndarray of shape (4,), type int, representing the index of the slice number where the organ starts
            organ_right: np.ndarray of shape (4,), type int, representing the index of the slice number where the organ ends
        """
        assert self.model_loaded, "Please load the model checkpoint first."
        self.load_data(ct_3d_volume_path, z_positions_path)
        min_slice_idx, max_slice_idx = self.find_min_max_slice_idxs()

        found_organs = np.full(shape=(4,), fill_value=False, dtype=bool)
        organ_left_bounds = np.full(shape=(4, 2), fill_value=min_slice_idx, dtype=np.int32)
        organ_right_bounds = np.full(shape=(4, 2), fill_value=max_slice_idx, dtype=np.int32)

        searched_slices = np.full(shape=max_slice_idx - min_slice_idx + 1, fill_value=False, dtype=bool)
        pred_suggestions = np.linspace(min_slice_idx, max_slice_idx, num=batch_size, dtype=np.int32)
        min_gap = max(int((max_slice_idx - min_slice_idx) * locate_organ_min_gap_ratio), 2)

        # loop
        while True:
            # predict at suggested locations
            preds = self.predict_at_locations(pred_suggestions)
            pred_slices = self.reduce_slice(preds) # shape (batch_size, 4)
            searched_slices[pred_suggestions - min_slice_idx] = True

            # update slices checking status
            for k in range(4):
                locations = np.argwhere(pred_slices[:, k]).squeeze(-1)
                if not found_organs[k]:
                    if np.any(pred_slices[:, k]):
                        found_organs[k] = True
                        organ_left_bounds[k, 0] = min_slice_idx
                        organ_left_bounds[k, 1] = pred_suggestions[locations[0]]
                        organ_right_bounds[k, 0] = pred_suggestions[locations[-1]]
                        organ_right_bounds[k, 1] = max_slice_idx
                else:
                    if np.any(pred_slices[:, k]):
                        organ_left_bounds[k, 1] = min(pred_suggestions[locations[0]], organ_left_bounds[k, 1])
                        organ_right_bounds[k, 0] = max(pred_suggestions[locations[-1]], organ_right_bounds[k, 0])
                    else:
                        slice_locations = pred_suggestions # all slices are predicted to be False
                        inside_left_bounds = (organ_left_bounds[k, 0] <= slice_locations) & (slice_locations < organ_left_bounds[k, 1])
                        inside_right_bounds = (organ_right_bounds[k, 0] < slice_locations) & (slice_locations <= organ_right_bounds[k, 1])
                        if np.any(inside_left_bounds):
                            # rightmost slice that is inside the left bound
                            organ_left_bounds[k, 0] = slice_locations[np.argwhere(inside_left_bounds).squeeze(-1)[-1]]
                        if np.any(inside_right_bounds):
                            # leftmost slice that is inside the right bound
                            organ_right_bounds[k, 1] = slice_locations[np.argwhere(inside_right_bounds).squeeze(-1)[0]]

            # make new predictions
            pred_suggestions = np.zeros(shape=(batch_size,), dtype=np.int32)
            idx = 0
            for k in range(4): # for each organ first
                if found_organs[k]: # if the organ is found, we search for the middle slice
                    if organ_left_bounds[k, 0] < organ_left_bounds[k, 1] - 1:
                        pred_suggestions[idx] = (organ_left_bounds[k, 0] + organ_left_bounds[k, 1]) // 2
                        idx += 1
                        if idx >= batch_size:
                            break
                    if organ_right_bounds[k, 0] < organ_right_bounds[k, 1] - 1:
                        pred_suggestions[idx] = (organ_right_bounds[k, 0] + organ_right_bounds[k, 1]) // 2
                        idx += 1
                        if idx >= batch_size:
                            break
            while idx < batch_size:
                # add the middle of maximum gaps between searched slices
                searched_locs = np.argwhere(searched_slices).squeeze(-1)
                gap_lengths = np.diff(searched_locs)
                max_gap_idx = np.argmax(gap_lengths)
                if gap_lengths[max_gap_idx] >= min_gap:
                    pred_suggestions[idx] = (searched_locs[max_gap_idx] + gap_lengths[max_gap_idx] // 2) + min_slice_idx
                    idx += 1
                else:
                    break
            # terminate if no more predictions are needed
            if idx == 0:
                break

        left = np.sum(organ_left_bounds, axis=-1) // 2
        right = np.sum(organ_right_bounds, axis=-1) // 2
        return found_organs, left, right

if __name__ == "__main__":
    folder = "stage1_organ_segmentator"
    model_folder = "models"
    if not os.path.exists(folder):
        os.makedirs(folder)

    import config
    import manager_folds
    import pandas as pd
    import shutil
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--channel_progression", type=int, nargs="+", default=[2, 3, 6, 9, 15, 32, 128, 256, 512, 1024],
                        help="The channels for progression in ResNet backbone.")
    parser.add_argument("--hidden3d_blocks", type=int, nargs="+", default=[1, 2, 1, 0, 0, 0],
                        help="Number of hidden 3d blocks for ResNet backbone.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 2, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--bottleneck_factor", type=int, default=4,
                        help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false",
                        help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    config.add_argparse_arguments(parser)

    args = parser.parse_args()
    config.parse_args(args)

    print("Getting dataset...")
    dataset = args.dataset
    dataset_entries = manager_folds.load_dataset(dataset)
    out_folder = os.path.join(folder, dataset)
    if os.path.isdir(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    segmentator = OrganSegmentator(channel_progression=args.channel_progression, res_conv3d_blocks=args.hidden3d_blocks,
                                        res_conv_blocks=args.hidden_blocks, bottleneck_factor=args.bottleneck_factor, squeeze_excitation=args.squeeze_excitation,
                                        device=config.device)
    print("Loading model...")
    segmentator.load_checkpoint(os.path.join(model_folder, args.model, "model.pt"))

    for patient_id in tqdm.tqdm(dataset_entries):
        patient_folder = os.path.join("data_hdf5_cropped", str(patient_id))
        for series_id in os.listdir(patient_folder):
            series_folder = os.path.join(patient_folder, series_id)

            ct_path = os.path.join(series_folder, "ct_3D_image.hdf5")
            z_pos_path = os.path.join(series_folder, "z_positions.npy")

            found_organs, left, right = segmentator.predict_organs(ct_path, z_pos_path)

            pd.DataFrame({
                "found": found_organs,
                "left": left,
                "right": right
            }, index=["liver", "spleen", "kidney", "bowel"]).to_csv(os.path.join(out_folder, series_id + ".csv"))
