import os
import multiprocessing

import h5py
import numpy as np
import torch
import torchvision.transforms.functional
import pandas as pd

import config
import image_ROI_sampler
import image_sampler_augmentations
import manager_stage1_results

def image_loading_subprocess(image_loading_pipe_recv, running: multiprocessing.Value,
                             max_slice_region_depth: int, max_image_width: int, max_image_height: int,
                             loaded_image_depth: multiprocessing.Value, loaded_image_width: multiprocessing.Value, loaded_image_height: multiprocessing.Value,
                             loaded_image_has_segmentations: multiprocessing.Value,
                             use_3d: bool,
                             image_available_lock: multiprocessing.Lock,
                             image_required_flag: multiprocessing.Value,
                             worker_name: str, buffer_max_size: 5):
    try:
        print("Subprocess {} starting...".format(worker_name))

        pending_images = []
        buffered_images = []

        image_shared_memory = multiprocessing.shared_memory.SharedMemory(create=False, name="{}_image".format(worker_name))
        image_shared_memory_array = np.ndarray((max_slice_region_depth, max_image_height, max_image_width), dtype=np.float32, buffer=image_shared_memory.buf)

        seg_shared_memory = multiprocessing.shared_memory.SharedMemory(create=False, name="{}_seg".format(worker_name))
        if use_3d:
            seg_shared_memory_array = np.ndarray((4, max_slice_region_depth, max_image_height, max_image_width), dtype=np.float32, buffer=seg_shared_memory.buf)
        else:
            seg_shared_memory_array = np.ndarray((4, max_image_height, max_image_width), dtype=np.float32, buffer=seg_shared_memory.buf)

        run_time = 0

        print("Subprocess started.".format(worker_name))
        while running.value:
            # fetch new requests into queue
            if image_loading_pipe_recv.poll():
                load_info = image_loading_pipe_recv.recv()
                pending_images.append(load_info)

            # if buffer not full, and pending images available, load new image into buffer
            if (len(buffered_images) < buffer_max_size) and (len(pending_images) > 0):
                load_info = pending_images.pop(0)
                # get info
                patient_id = str(load_info["patient_id"])
                series_id = str(load_info["series_id"])
                segmentation_folder = str(load_info["segmentation_folder"])
                slice_locs = np.array(load_info["slices"])
                slice_center = load_info["slice_center"]
                elastic_augmentation = load_info["elastic_augmentation"]
                load_segmentations = load_info["load_segmentations"]

                # load image and apply augmentation if needed
                with h5py.File(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "ct_3D_image.hdf5"), "r") as f:
                    image_slice = image_ROI_sampler.load_slice_from_hdf(f, slice_locs, "ct_3D_image")
                image_slice = torch.from_numpy(image_slice)
                original_height = image_slice.shape[1]
                original_width = image_slice.shape[2]

                if not load_segmentations:
                    segmentation_slice = None
                else:
                    with h5py.File(os.path.join(segmentation_folder, str(series_id) + ".hdf5"), "r") as f:
                        if not use_3d:
                            segmentation_3D_image = f["segmentation_arr"]
                            segmentation_raw = segmentation_3D_image[slice_center, ...].astype(dtype=bool)
                            segmentation_slice = np.zeros((original_height, original_width, 4), dtype=bool)
                            segmentation_slice[..., :2] = segmentation_raw[..., :2]
                            segmentation_slice[..., 2] = np.any(segmentation_raw[..., 2:4], axis=-1)
                            segmentation_slice[..., 3] = segmentation_raw[..., 4]
                        else:
                            segmentation_raw = image_ROI_sampler.load_slice_from_hdf(f, slice_locs, "segmentation_arr").astype(dtype=bool)
                            segmentation_slice = np.zeros((len(slice_locs), original_height, original_width, 4),
                                                          dtype=bool)
                            segmentation_slice[..., :2] = segmentation_raw[..., :2]
                            segmentation_slice[..., 2] = np.any(segmentation_raw[..., 2:4], axis=-1)
                            segmentation_slice[..., 3] = segmentation_raw[..., 4]
                        del segmentation_raw
                    if not use_3d:
                        segmentation_slice = torch.tensor(segmentation_slice, dtype=torch.float32).permute((2, 0, 1))
                    else:
                        segmentation_slice = torch.tensor(segmentation_slice, dtype=torch.float32).permute((3, 0, 1, 2))

                # apply elastic deformation to height width. this is False always, but code is kept here for backwards compatibility with CPU version
                if elastic_augmentation:
                    # 3d elastic deformation (varying 2d elastic deformation over depth)
                    displacement_field = image_sampler_augmentations.generate_displacement_field3D(original_width, original_height,
                                                                                        max_slice_region_depth, [0.3, 0.7, 1, 0.7, 0.3])
                    displacement_field = torch.from_numpy(displacement_field)
                    image_slice = image_sampler_augmentations.apply_displacement_field3D(image_slice, displacement_field)
                    if load_segmentations:
                        if use_3d:
                            segmentation_slice = image_sampler_augmentations.apply_displacement_field3D(segmentation_slice, displacement_field)
                        else:  # apply deformation in center slice only
                            segmentation_slice = image_sampler_augmentations.apply_displacement_field(
                                segmentation_slice,
                                displacement_field[(displacement_field.shape[0] - 1) // 2, ...].unsqueeze(0))

                buffered_images.append({"image": image_slice, "segmentation": segmentation_slice})

            # if buffer not empty, and image required, load image from buffer
            if image_required_flag.value and (len(buffered_images) > 0):
                # set flag false
                image_required_flag.value = False

                # check whether segmentation is required
                image_data = buffered_images.pop(0)
                has_segmentation = image_data["segmentation"] is not None

                # place data into shared memory
                img_d, img_h, img_w = image_data["image"].shape
                image_shared_memory_array[:img_d, :img_h, :img_w] = image_data["image"].numpy()
                if has_segmentation:
                    if use_3d:
                        seg_shared_memory_array[:, :img_d, :img_h, :img_w] = image_data["segmentation"].numpy()
                    else:
                        seg_shared_memory_array[:, :img_h, :img_w] = image_data["segmentation"].numpy()

                # set value to pass to main process
                loaded_image_depth.value = img_d
                loaded_image_height.value = img_h
                loaded_image_width.value = img_w
                loaded_image_has_segmentations.value = has_segmentation

                # release lock
                image_available_lock.release()

            time.sleep(0.003)

            run_time += 1
            if run_time % 10000 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("Subprocess interrupted....")

    image_shared_memory.close()
    seg_shared_memory.close()
    print("Subprocess terminated.")

class SliceLoaderWorker:
    def __init__(self, worker_name: str, max_slice_region_depth=9, use_3d=False, max_image_width=576, max_image_height=512):
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height
        self.use_3d = use_3d
        image_loading_pipe_recv, self.image_loading_pipe_send = multiprocessing.Pipe(duplex=False)
        self.running = multiprocessing.Value(ctypes.c_bool, True)
        self.image_available_lock = multiprocessing.Lock()
        self.image_required_flag = multiprocessing.Value(ctypes.c_bool, True)
        self.image_available_lock.acquire(block=True)

        img = np.zeros((max_slice_region_depth, max_image_height, max_image_width), dtype=np.float32)
        self.image_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=img.nbytes, name="{}_image".format(worker_name))
        del img
        self.image_shared_memory_array = np.ndarray((max_slice_region_depth, max_image_height, max_image_width), dtype=np.float32, buffer=self.image_shared_memory.buf)

        if use_3d:
            seg = np.zeros((4, max_slice_region_depth, max_image_height, max_image_width), dtype=np.float32)
            self.seg_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=seg.nbytes,
                                                                                  name="{}_seg".format(worker_name))
            del seg
            self.seg_shared_memory_array = np.ndarray((4, max_slice_region_depth, max_image_height, max_image_width),
                                                        dtype=np.float32, buffer=self.seg_shared_memory.buf)
        else:
            seg = np.zeros((4, max_image_height, max_image_width), dtype=np.float32)
            self.seg_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=seg.nbytes,
                                                                                name="{}_seg".format(worker_name))
            del seg
            self.seg_shared_memory_array = np.ndarray((4, max_image_height, max_image_width),
                                                      dtype=np.float32, buffer=self.seg_shared_memory.buf)

        self.loaded_image_depth = multiprocessing.Value(ctypes.c_int, 0)
        self.loaded_image_width = multiprocessing.Value(ctypes.c_int, 0)
        self.loaded_image_height = multiprocessing.Value(ctypes.c_int, 0)
        self.loaded_image_has_segmentations = multiprocessing.Value(ctypes.c_bool, False)

        self.process = multiprocessing.Process(target=image_loading_subprocess, args=(image_loading_pipe_recv, self.running,
                                                                                        max_slice_region_depth, max_image_width, max_image_height,
                                                                                        self.loaded_image_depth, self.loaded_image_width, self.loaded_image_height,
                                                                                        self.loaded_image_has_segmentations,
                                                                                        use_3d,
                                                                                        self.image_available_lock,
                                                                                        self.image_required_flag, worker_name, 5))
        self.process.start()

    def terminate(self):
        self.running.value = False
        self.process.join()
        self.image_shared_memory.close()
        self.image_shared_memory.unlink()
        self.seg_shared_memory.close()
        self.seg_shared_memory.unlink()

    def request_load_image(self, patient_id, series_id, segmentation_folder: str, slices: list[int], slice_center: int, elastic_augmentation: bool, load_segmentations: bool):
        self.image_loading_pipe_send.send({
            "patient_id": patient_id,
            "series_id": series_id,
            "segmentation_folder": segmentation_folder,
            "slices": slices,
            "slice_center": slice_center,
            "elastic_augmentation": elastic_augmentation,
            "load_segmentations": load_segmentations
        })

    def get_requested_image(self):
        self.image_required_flag.value = True
        self.image_available_lock.acquire(block=True)

        if self.loaded_image_has_segmentations.value:
            if self.use_3d:
                return self.image_shared_memory_array[:self.loaded_image_depth.value, :self.loaded_image_height.value, :self.loaded_image_width.value].copy(),\
                        self.seg_shared_memory_array[:, :self.loaded_image_depth.value, :self.loaded_image_height.value, :self.loaded_image_width.value].copy()
            else:
                return self.image_shared_memory_array[:self.loaded_image_depth.value, :self.loaded_image_height.value, :self.loaded_image_width.value].copy(),\
                        self.seg_shared_memory_array[:, :self.loaded_image_height.value, :self.loaded_image_width.value].copy()
        else:
            return self.image_shared_memory_array[:self.loaded_image_depth.value, :self.loaded_image_height.value, :self.loaded_image_width.value].copy(), \
                     None

def load_image(patient_ids: list,
               series_ids: list,
               organ_id: int, organ_height: int, organ_width: int,  # organ id, expected organ height and width
               stage1_information: manager_stage1_results.Stage1ResultsManager,
               organ_sampling_depth=9,
               translate_rotate_augmentation=False,
               elastic_augmentation=False) -> (torch.Tensor, torch.Tensor, np.ndarray):
    assert len(patient_ids) == len(series_ids), "patient_ids and series_ids must have the same length"
    batch_size = len(patient_ids)

    ## Compute the required height and width
    max_angle = 15 * np.pi / 180
    cur_angle = np.arctan2(organ_height, organ_width)
    diag = np.hypot(organ_height + 64.0, organ_width + 64.0)
    req_rot_w = int(np.ceil(diag * max(np.sin(cur_angle + max_angle), np.sin(cur_angle - max_angle))))
    req_rot_h = int(np.ceil(diag * max(np.cos(cur_angle + max_angle), np.cos(cur_angle - max_angle))))

    ## Load the images
    image_batch = torch.zeros((batch_size, 1, organ_sampling_depth, req_rot_h, req_rot_w), dtype=torch.float32,
                              device=config.device)
    organ_loc_batch = torch.zeros((batch_size, 1, req_rot_h, req_rot_w), dtype=torch.float32,
                                  device=config.device)
    for k in range(batch_size):
        organ_slice_min, organ_slice_max = stage1_information.get_organ_slicelocs(int(series_ids[k]), organ_id)
        image, organ_location = load_series_image_and_organloc_from_minmax(str(patient_ids[k]), str(series_ids[k]),
                                                                           organ_id, organ_sampling_depth,
                                                                           organ_slice_min, organ_slice_max,
                                                                           req_rot_w, req_rot_h,
                                                                           stage1_information.segmentation_dataset_folder,
                                                                           elastic_augmentation)
        image_batch[k, 1, ...].copy_(torch.from_numpy(image), non_blocking=True)
        organ_loc_batch[k, 1, ...].copy_(torch.from_numpy(organ_location), non_blocking=True)

    ## Apply elastic deformation to height width
    if elastic_augmentation:
        # 3d elastic deformation (varying 2d elastic deformation over depth), and also varying over slices
        displacement_field = torch.stack([image_sampler_augmentations.generate_displacement_field3D(req_rot_w,
                                                                                                    req_rot_h,
                                                                                                    organ_sampling_depth,
                                                                                                    # (batch_size, organ_sampling_depth, H, W, 2)
                                                                                                    kernel_depth_span=[
                                                                                                        0.3, 0.7, 1,
                                                                                                        0.7, 0.3],
                                                                                                    device=config.device)
                                          for k in range(batch_size)], dim=0)
        assert displacement_field.shape == (batch_size, organ_sampling_depth, req_rot_h, req_rot_w, 2)
        image_batch = image_sampler_augmentations.apply_displacement_field3D_simple(
            image_batch.reshape(batch_size * organ_sampling_depth, 1, req_rot_h, req_rot_w),
            displacement_field.view(batch_size * organ_sampling_depth, req_rot_h, req_rot_w, 2)) \
            .view(batch_size, 1, organ_sampling_depth, req_rot_h, req_rot_w)
        organ_loc_batch = torch.any(image_sampler_augmentations.apply_displacement_field3D_simple(
            organ_loc_batch.expand(batch_size, organ_sampling_depth, req_rot_h, req_rot_w)
            .reshape(batch_size * organ_sampling_depth, 1, req_rot_h, req_rot_w),
            displacement_field.view(batch_size * organ_sampling_depth, req_rot_h, req_rot_w, 2))
                                    .view(batch_size, 1, organ_sampling_depth, req_rot_h, req_rot_w) > 0.5,
                                    dim=2).float()

    ## Apply rotation augmentation to height width
    if translate_rotate_augmentation:
        # Generate the rotation angles
        rotation_angles = np.random.uniform(low=-max_angle, high=max_angle, size=batch_size)
        rotation_angles = rotation_angles * 180.0 / np.pi  # convert to degrees

        # Rotate the image
        image_batch = image_sampler_augmentations.rotate(image_batch, list(rotation_angles))
        organ_loc_batch = image_sampler_augmentations.rotate(
            organ_loc_batch.view(batch_size, 1, 1, req_rot_h, req_rot_w), list(rotation_angles)) \
                              .view(batch_size, req_rot_h, req_rot_w) > 0.5

    ## Crop the image to the desired size, and apply translation augmentation if necessary
    final_image_batch = torch.zeros((batch_size, 1, organ_sampling_depth, organ_height, organ_width),
                                    dtype=torch.float32,
                                    device=config.device)
    for k in range(batch_size):
        # compute organ bounds
        heights = torch.any(organ_loc_batch[k, ...], dim=-1)
        widths = torch.any(organ_loc_batch[k, ...], dim=-2)
        heights = np.argwhere(heights.cpu().numpy())
        widths = np.argwhere(widths.cpu().numpy())
        if len(heights) == 0 or len(widths) == 0:
            mid_x = req_rot_h // 2
            mid_y = req_rot_w // 2
        else:
            mid_x = (np.min(widths) + np.max(widths)) // 2
            mid_y = (np.min(heights) + np.max(heights)) // 2
        x_min, x_max = mid_x - organ_width // 2, mid_x + organ_width // 2
        y_min, y_max = mid_y - organ_height // 2, mid_y + organ_height // 2

        # correct for out of bounds
        if x_min < 0:
            x_max -= x_min
            x_min = 0
        elif x_max >= req_rot_w:
            x_min -= (x_max - req_rot_w + 1)
            x_max = req_rot_w - 1
        if y_min < 0:
            y_max -= y_min
            y_min = 0
        elif y_max >= req_rot_h:
            y_min -= (y_max - req_rot_h + 1)
            y_max = req_rot_h - 1

        # apply translation augmentation
        if translate_rotate_augmentation:
            left_available, right_available = x_min, req_rot_w - x_max - 1
            top_available, bottom_available = y_min, req_rot_h - y_max - 1
            x_translation = np.random.randint(min(left_available, 48), min(right_available + 1, 48))
            y_translation = np.random.randint(min(top_available, 48), min(bottom_available + 1, 48))
            x_min, x_max = x_min + x_translation, x_max + x_translation
            y_min, y_max = y_min + y_translation, y_max + y_translation

        # crop the image
        final_image_batch[k, 0, ...].copy_(image_batch[k, 0, :, y_min:y_max + 1, x_min:x_max + 1], non_blocking=True)

    return final_image_batch
