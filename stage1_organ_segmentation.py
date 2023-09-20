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

def consecutive_repeats(arr):
    if len(arr) == 0:
        return np.array([])
    else:
        diff = np.diff(arr)
        idx = np.argwhere(diff != 0).squeeze(-1) + 1
        idx = np.concatenate([np.array([0]), idx, np.array([len(arr)])], axis=0)
        repeats = np.diff(idx)
        return arr[idx[:-1]], repeats

def model_predict(model, image_tensor):
    with torch.no_grad():
        pred = model(image_tensor) > 0
    return pred

model_predict_compile = torch.compile(model_predict)

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
        self.data_is_cached = False
        self.loaded_data_H = None
        self.loaded_data_W = None

    def load_checkpoint(self, model_checkpoint_path: str):
        # for example, model_checkpoint_path = "/path/to/model.pt"
        self.model.load_state_dict(torch.load(model_checkpoint_path))
        self.model.eval()
        self.model_loaded = True

    def load_data(self, ct_3d_volume_path: str, z_positions_path: str, cache_data = True):
        if self.data_loaded:
            self.close()
        self.ct_3d_volume = h5py.File(ct_3d_volume_path, "r")
        self.z_positions = np.load(z_positions_path)
        self.is_flipped = np.diff(self.z_positions).mean() > 0.0
        self.data_loaded = True
        self.loaded_data_H = self.ct_3d_volume["ct_3D_image"].shape[-2]
        self.loaded_data_W = self.ct_3d_volume["ct_3D_image"].shape[-1]

        self.data_is_cached = cache_data
        if cache_data:
            ct_3d_volume = self.ct_3d_volume["ct_3D_image"][()]
            self.ct_3d_volume.close()
            self.ct_3d_volume = ct_3d_volume

    def close(self):
        if self.data_loaded:
            if not self.data_is_cached:
                self.ct_3d_volume.close()
            self.ct_3d_volume = None
            self.z_positions = None
            self.data_loaded = False

    def get_resize_strategy(self):
        assert self.data_loaded, "Please load the data first."
        resize_strategy = {}
        if self.loaded_data_H > 512:
            resize_strategy["H"] = {
                "mode": "crop",
                "start": (self.loaded_data_H - 512) // 2,
                "end": (self.loaded_data_H + 512) // 2
            }
        else:
            pad_top = (512 - self.loaded_data_H) // 2
            pad_bottom = 512 - self.loaded_data_H - pad_top
            resize_strategy["H"] = {
                "mode": "pad",
                "start": pad_top,
                "end": pad_bottom
            }
        if self.loaded_data_W > 576:
            resize_strategy["W"] = {
                "mode": "crop",
                "start": (self.loaded_data_W - 576) // 2,
                "end": (self.loaded_data_W + 576) // 2
            }
        else:
            pad_left = (576 - self.loaded_data_W) // 2
            pad_right = 576 - self.loaded_data_W - pad_left
            resize_strategy["W"] = {
                "mode": "pad",
                "start": pad_left,
                "end": pad_right
            }
        return resize_strategy

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

    def get_nearest_slice_indices(self, slice_idx, stride_mm: int=5, depth: int=9):
        """
        Same as the implementation in image_ROI_sampler. This is included here to
        avoid importing the other packages used by image_ROI_sampler for shipping.
        """
        assert self.data_loaded, "Please load the data first."
        assert depth % 2 == 1, "depth must be an odd number"
        depth_radius = (depth - 1) // 2

        current_zpos = self.z_positions[slice_idx]
        expected_zposes = np.arange(current_zpos - depth_radius * stride_mm,
                                    current_zpos + (depth_radius + 1) * stride_mm, stride_mm)

        # find nearest slice indices for the given z positions
        if self.is_flipped:
            nearest_slice_indices = find_closest(self.z_positions, expected_zposes)
        else:
            nearest_slice_indices = (self.z_positions.shape[0] - 1 -
                                     find_closest(self.z_positions[::-1], expected_zposes))[::-1]
        nearest_slice_indices[depth_radius] = slice_idx
        return nearest_slice_indices

    def load_local_features(self, slice_idx, stride_mm: int=5, depth: int=9):
        nearest_slice_indices = self.get_nearest_slice_indices(slice_idx, stride_mm, depth)
        nearest_slice_indices = np.clip(nearest_slice_indices, 0, self.z_positions.shape[0] - 1)

        # load from the h5py file
        collapsed_nearest_indices, repeats = consecutive_repeats(nearest_slice_indices)
        local_slice_image = self.ct_3d_volume["ct_3D_image"][collapsed_nearest_indices, ...]
        local_slice_image = np.repeat(local_slice_image, repeats, axis=0)
        if self.is_flipped:
            local_slice_image = local_slice_image[::-1, ...].copy()
        return local_slice_image

    def load_image_from_slice_indices(self, slice_idxs: np.ndarray, stride_mm: int=5, depth: int=9):
        if self.data_is_cached:
            indices = np.stack([np.clip(self.get_nearest_slice_indices(slice_idxs[k], stride_mm, depth), 0,
                                        self.z_positions.shape[0] - 1)
                                for k in range(len(slice_idxs))], axis=0) # (batch_size, depth)
            assert indices.shape == (len(slice_idxs), depth)
            batch_volume = self.ct_3d_volume[np.expand_dims(indices, axis=1), ...]
            assert batch_volume.shape[:3] == (len(slice_idxs), 1, depth)
            return torch.tensor(batch_volume, dtype=torch.float32, device=self.device)
        else:
            return torch.stack([
                torch.tensor(self.load_local_features(slice_idxs[k], stride_mm, depth), dtype=torch.float32,
                             device=self.device) for k in range(len(slice_idxs))], dim=0).unsqueeze(1)

    def predict_at_locations(self, slice_idxs: np.ndarray, stride_mm: int=5, depth: int=9):
        assert self.model_loaded, "Please load the model checkpoint first."
        assert self.data_loaded, "Please load the data first."
        assert depth % 2 == 1, "depth must be an odd number"

        image_tensor = self.load_image_from_slice_indices(slice_idxs, stride_mm, depth)
        # the shape should be (batch_size, 1, D, H, W)
        assert image_tensor.shape == (len(slice_idxs), 1, depth, self.loaded_data_H, self.loaded_data_W)

        resize_strategy = self.get_resize_strategy()
        with torch.no_grad():
            if resize_strategy["H"]["mode"] == "crop":
                image_tensor = image_tensor[..., resize_strategy["H"]["start"]:resize_strategy["H"]["end"], :]
            else:
                image_tensor = torch.nn.functional.pad(image_tensor, (0, 0, resize_strategy["H"]["start"], resize_strategy["H"]["end"]))
            if resize_strategy["W"]["mode"] == "crop":
                image_tensor = image_tensor[..., resize_strategy["W"]["start"]:resize_strategy["W"]["end"]]
            else:
                image_tensor = torch.nn.functional.pad(image_tensor, (resize_strategy["W"]["start"], resize_strategy["W"]["end"]))
            assert image_tensor.shape == (len(slice_idxs), 1, depth, 512, 576)

            preds = model_predict_compile(self.model, image_tensor)

        # the shape should be (batch_size, 4, H, W)
        assert preds.shape[:2] == (len(slice_idxs), 4)
        return preds

    @staticmethod
    def reduce_slice(preds):
        with torch.no_grad():
            return torch.any(torch.any(preds, -1), -1).cpu().numpy()


    def predict_organs(self, ct_3d_volume_path: str, z_positions_path: str, batch_size=32,
                            locate_organ_min_gap_ratio=0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            organ_location: np.ndarray of shape (4, H, W), type int, representing the rough prediction of location of the organ
        """
        assert self.model_loaded, "Please load the model checkpoint first."
        assert batch_size >= 8, "batch_size must be at least 8"
        self.load_data(ct_3d_volume_path, z_positions_path)
        # min_slice_idx, max_slice_idx = self.find_min_max_slice_idxs()
        min_slice_idx, max_slice_idx = 0, self.z_positions.shape[0] - 1 # use all slices
        # the variables below are used to store the results
        found_organs = np.full(shape=(4,), fill_value=False, dtype=bool)
        organ_left_bounds = np.full(shape=(4, 2), fill_value=min_slice_idx, dtype=np.int32)
        organ_right_bounds = np.full(shape=(4, 2), fill_value=max_slice_idx, dtype=np.int32)
        # the states of the search algorithm
        searched_slices = np.full(shape=max_slice_idx - min_slice_idx + 1, fill_value=False, dtype=bool)
        pred_suggestions = np.linspace(min_slice_idx, max_slice_idx, num=batch_size, dtype=np.int32)
        min_gap = max(int((max_slice_idx - min_slice_idx) * locate_organ_min_gap_ratio), 2)
        # the model 2D location of the organs
        organ_location = torch.zeros(size=(4, 16, 18), dtype=torch.bool, device=self.device)

        # loop
        while True:
            # predict at suggested locations
            preds = self.predict_at_locations(pred_suggestions) # shape (batch_size, 4, 16, 18)
            pred_slices = self.reduce_slice(preds) # shape (batch_size, 4)

            searched_slices[pred_suggestions - min_slice_idx] = True
            # update organ_location
            with torch.no_grad():
                organ_location = torch.logical_or(torch.any(preds, dim=0), organ_location)

            # update slices checking status
            for k in range(4):
                if not found_organs[k]:
                    locations = np.argwhere(pred_slices[:, k]).squeeze(-1)
                    if np.any(pred_slices[:, k]):
                        found_organs[k] = True
                        organ_left_bounds[k, 0] = min_slice_idx
                        organ_left_bounds[k, 1] = pred_suggestions[locations[0]]
                        organ_right_bounds[k, 0] = pred_suggestions[locations[-1]]
                        organ_right_bounds[k, 1] = max_slice_idx
                else:
                    # boolean mask of which slices are in the left and right bounds
                    inside_left_bounds = (organ_left_bounds[k, 0] <= pred_suggestions) & (pred_suggestions < organ_left_bounds[k, 1])
                    inside_right_bounds = (organ_right_bounds[k, 0] < pred_suggestions) & (pred_suggestions <= organ_right_bounds[k, 1])

                    if inside_left_bounds.sum() > 0:
                        pred_left_slices = pred_slices[inside_left_bounds, k]
                        pred_left_suggestions = pred_suggestions[inside_left_bounds]
                        if np.any(pred_left_slices):
                            # the right bound of the left bounds will be set to the leftmost location of detected organ
                            locations = np.argwhere(pred_left_slices).squeeze(-1)
                            organ_left_bounds[k, 1] = min(pred_left_suggestions[locations[0]], organ_left_bounds[k, 1])
                        else:
                            # the left bound of the left bounds will be set to the rightmost proposed locations
                            organ_left_bounds[k, 0] = max(pred_left_suggestions[-1], organ_left_bounds[k, 0])
                    if inside_right_bounds.sum() > 0:
                        pred_right_slices = pred_slices[inside_right_bounds, k]
                        pred_right_suggestions = pred_suggestions[inside_right_bounds]
                        if np.any(pred_right_slices):
                            # the left bound of the right bounds will be set to the rightmost location of detected organ
                            locations = np.argwhere(pred_right_slices).squeeze(-1)
                            organ_right_bounds[k, 0] = max(pred_right_suggestions[locations[-1]], organ_right_bounds[k, 0])
                        else:
                            # the right bound of the right bounds will be set to the leftmost proposed locations
                            organ_right_bounds[k, 1] = min(pred_right_suggestions[0], organ_right_bounds[k, 1])

            # make new predictions
            pred_suggestions = np.full(shape=(batch_size,), fill_value=-1, dtype=np.int32)
            completed_organs = found_organs & (organ_left_bounds[:, 0] >= organ_left_bounds[:, 1] - 1) & (organ_right_bounds[:, 0] >= organ_right_bounds[:, 1] - 1)
            num_organs_to_predict = np.sum(~completed_organs)

            allocate_prediction = np.linspace(0, batch_size, num=num_organs_to_predict + 1, dtype=np.int32)
            organ_alloc = 0

            for k in range(4): # for each organ, if not completed
                if completed_organs[k]:
                    continue
                # allocation of indices to predict for the organ
                organ_alloc_min, organ_alloc_max = allocate_prediction[organ_alloc], allocate_prediction[organ_alloc + 1]
                organ_alloc += 1

                if found_organs[k]: # if the organ is found, we search for the middle slice
                    left_required = organ_left_bounds[k, 0] < organ_left_bounds[k, 1] - 1
                    right_required = organ_right_bounds[k, 0] < organ_right_bounds[k, 1] - 1
                    if left_required and right_required:
                        alloc_mid_split = (organ_alloc_min + organ_alloc_max) // 2
                        pred_suggestions[organ_alloc_min:alloc_mid_split] = np.linspace(organ_left_bounds[k, 0], organ_left_bounds[k, 1],
                                                                                        num=alloc_mid_split - organ_alloc_min + 2, dtype=np.int32)[1:-1]
                        pred_suggestions[alloc_mid_split:organ_alloc_max] = np.linspace(organ_right_bounds[k, 0], organ_right_bounds[k, 1],
                                                                                        num=organ_alloc_max - alloc_mid_split + 2, dtype=np.int32)[1:-1]
                    elif left_required:
                        pred_suggestions[organ_alloc_min:organ_alloc_max] = np.linspace(organ_left_bounds[k, 0], organ_left_bounds[k, 1],
                                                                                        num=organ_alloc_max - organ_alloc_min + 2, dtype=np.int32)[1:-1]
                    elif right_required:
                        pred_suggestions[organ_alloc_min:organ_alloc_max] = np.linspace(organ_right_bounds[k, 0], organ_right_bounds[k, 1],
                                                                                        num=organ_alloc_max - organ_alloc_min + 2, dtype=np.int32)[1:-1]
                else:
                    # if the organ is not found, we add the middle of the maximum gaps between searched slices
                    searched_locs = np.argwhere(searched_slices).squeeze(-1)
                    gap_lengths = np.diff(searched_locs)
                    max_gap_idx = np.argmax(gap_lengths)
                    if gap_lengths[max_gap_idx] >= min_gap:
                        pred_suggestions[organ_alloc_min:organ_alloc_max] = np.linspace(searched_locs[max_gap_idx], searched_locs[max_gap_idx]
                                                                                        + gap_lengths[max_gap_idx], num=organ_alloc_max - organ_alloc_min + 2,
                                                                                        dtype=np.int32)[1:-1] + min_slice_idx
                searched_slices[pred_suggestions[organ_alloc_min:organ_alloc_max] - min_slice_idx] = True
            # terminate if no more predictions are needed
            suggest_locs = pred_suggestions != -1
            if suggest_locs.sum() == 0:
                break
            pred_suggestions = np.unique(pred_suggestions[suggest_locs])

        left = np.sum(organ_left_bounds, axis=-1) // 2
        right = np.sum(organ_right_bounds, axis=-1) // 2

        organ_location = organ_location.cpu().numpy()
        # upscale to size fed into model
        organ_location = np.repeat(np.repeat(organ_location, repeats=32, axis=1), repeats=32, axis=2)
        # crop to original size by reversing the resize strategy
        resize_strategy = self.get_resize_strategy()
        if resize_strategy["H"]["mode"] == "crop":
            pad_top = resize_strategy["H"]["start"]
            pad_bottom = self.loaded_data_H - resize_strategy["H"]["end"]
            organ_location = np.pad(organ_location, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=False)
        else:
            organ_location = organ_location[:, resize_strategy["H"]["start"]:-resize_strategy["H"]["end"], :]
        if resize_strategy["W"]["mode"] == "crop":
            pad_left = resize_strategy["W"]["start"]
            pad_right = self.loaded_data_W - resize_strategy["W"]["end"]
            organ_location = np.pad(organ_location, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant", constant_values=False)
        else:
            organ_location = organ_location[:, :, resize_strategy["W"]["start"]:-resize_strategy["W"]["end"]]

        assert organ_location.shape == (4, self.loaded_data_H, self.loaded_data_W)
        return found_organs, left, right, organ_location

def predict_organ_location(self, found_organs1, left1, right1, z_poses1,
                            mask_organs2, left2, right2, z_poses2):
    """
    Predicts the z-positions of the organs in the second image, based on the organs
    in the first image, and selected organs in the second image.
    """
    pass

if __name__ == "__main__":
    model_folder = "models"

    import config
    import manager_folds
    import manager_stage1_results
    import pandas as pd
    import shutil
    import tqdm

    folder = manager_stage1_results.SEGMENTATION_RESULTS_FOLDER
    if not os.path.exists(folder):
        os.makedirs(folder)

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

            found_organs, left, right, organ_location = segmentator.predict_organs(ct_path, z_pos_path)

            pd.DataFrame({
                "found": found_organs,
                "left": left,
                "right": right
            }, index=["liver", "spleen", "kidney", "bowel"]).to_csv(os.path.join(out_folder, series_id + ".csv"))

            with h5py.File(os.path.join(out_folder, series_id + ".hdf5"), "w") as f:
                f.create_dataset("organ_location", data=organ_location, compression="gzip", compression_opts=3)
