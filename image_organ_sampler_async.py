import os
import time
import gc
import ctypes
import multiprocessing
import multiprocessing.shared_memory

import h5py
import numpy as np
import torch
import torchvision.transforms.functional
import pandas as pd

import config
import image_organ_sampler
import image_sampler_augmentations
import manager_stage1_results

def image_loading_subprocess(image_loading_pipe_recv, running: multiprocessing.Value,
                             organ_sampling_depth: int, organ_id: int,
                             target_width: int, target_height: int,
                             image_available_lock: multiprocessing.Lock,
                             image_required_flag: multiprocessing.Value,
                             has_segmentation_flag: multiprocessing.Value,
                             worker_name: str, buffer_max_size: 5):
    try:
        print("Subprocess {} starting...".format(worker_name))

        pending_images = []
        buffered_images = []

        image_shared_memory = multiprocessing.shared_memory.SharedMemory(create=False, name="{}_image".format(worker_name))
        image_shared_memory_array = np.ndarray((organ_sampling_depth, target_height, target_width), dtype=np.float32, buffer=image_shared_memory.buf)

        organ_segmentation_shared_memory = multiprocessing.shared_memory.SharedMemory(create=False, name="{}_organ_seg".format(worker_name))
        organ_segmentation_shared_memory_array = np.ndarray((organ_sampling_depth, target_height, target_width), dtype=np.float32, buffer=organ_segmentation_shared_memory.buf)

        organ_loc_shared_memory = multiprocessing.shared_memory.SharedMemory(create=False, name="{}_organ".format(worker_name))
        organ_loc_shared_memory_array = np.ndarray((target_height, target_width), dtype=bool, buffer=image_shared_memory.buf)

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
                min_slice = int(load_info["min_slice"])
                max_slice = int(load_info["max_slice"])
                segmentation_dataset_folder = str(load_info["segmentation_dataset_folder"])
                data_hdf5_cropped_folder = str(load_info["data_hdf5_cropped_folder"])
                elastic_augmentation = bool(load_info["elastic_augmentation"])
                load_perslice_segmentation = bool(load_info["load_perslice_segmentation"])

                image, organ_location, organ_segmentation = image_organ_sampler.load_series_image_and_organloc_from_minmax(
                    patient_id,
                    series_id,
                    organ_id, organ_sampling_depth,
                    min_slice, max_slice,
                    target_width, target_height,
                    segmentation_dataset_folder,
                    data_hdf5_cropped_folder,
                    elastic_augmentation,
                    load_perslice_segmentation)

                buffered_images.append({"image": image, "organ_location": organ_location, "organ_segmentation": organ_segmentation})

            # if buffer not empty, and image required, load image from buffer
            if image_required_flag.value and (len(buffered_images) > 0):
                # set flag false
                image_required_flag.value = False

                # place data into shared memory
                image_data = buffered_images.pop(0)
                image_shared_memory_array[:, :, :] = image_data["image"]
                organ_loc_shared_memory_array[:, :] = image_data["organ_location"]
                has_segmentation_flag.value = image_data["organ_segmentation"] is not None
                if has_segmentation_flag.value:
                    organ_segmentation_shared_memory_array[:, :, :] = image_data["organ_segmentation"]

                # release lock
                image_available_lock.release()

            time.sleep(0.003)

            run_time += 1
            if run_time % 10000 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("Subprocess interrupted....")

    image_shared_memory.close()
    organ_segmentation_shared_memory.close()
    organ_loc_shared_memory.close()
    print("Subprocess terminated.")

class SliceLoaderWorker:
    def __init__(self, worker_name: str, organ_sampling_depth: int, organ_id: int,
                                        target_width: int, target_height: int):
        self.organ_sampling_depth = organ_sampling_depth
        self.organ_id = organ_id
        self.target_width = target_width
        self.target_height = target_height

        image_loading_pipe_recv, self.image_loading_pipe_send = multiprocessing.Pipe(duplex=False)
        self.running = multiprocessing.Value(ctypes.c_bool, True)
        self.image_available_lock = multiprocessing.Lock()
        self.image_required_flag = multiprocessing.Value(ctypes.c_bool, True)
        self.image_available_lock.acquire(block=True)

        img = np.zeros((organ_sampling_depth, target_height, target_width), dtype=np.float32)
        self.image_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=img.nbytes, name="{}_image".format(worker_name))
        del img
        self.image_shared_memory_array = np.ndarray((organ_sampling_depth, target_height, target_width), dtype=np.float32, buffer=self.image_shared_memory.buf)

        organ_slice_loc = np.zeros((organ_sampling_depth, target_height, target_width), dtype=np.float32)
        self.organ_segmentation_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=organ_slice_loc.nbytes, name="{}_organ_seg".format(worker_name))
        del organ_slice_loc
        self.organ_segmentation_shared_memory_array = np.ndarray((organ_sampling_depth, target_height, target_width), dtype=np.float32, buffer=self.organ_segmentation_shared_memory.buf)

        organ_loc = np.zeros((target_height, target_width), dtype=bool)
        self.organ_loc_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=organ_loc.nbytes,
                                                                              name="{}_organ".format(worker_name))
        del organ_loc
        self.organ_loc_shared_memory_array = np.ndarray((target_height, target_width),
                                                    dtype=bool, buffer=self.organ_loc_shared_memory.buf)

        self.has_segmentation_flag = multiprocessing.Value(ctypes.c_bool, False)

        self.process = multiprocessing.Process(target=image_loading_subprocess,
                                               args=(image_loading_pipe_recv, self.running,
                                                    organ_sampling_depth, organ_id,
                                                    target_width, target_height,
                                                    self.image_available_lock,
                                                    self.image_required_flag, self.has_segmentation_flag,
                                                     worker_name, 5))
        self.process.start()

    def terminate(self):
        self.running.value = False
        self.process.join()
        self.image_shared_memory.close()
        self.image_shared_memory.unlink()
        self.organ_segmentation_shared_memory.close()
        self.organ_segmentation_shared_memory.unlink()
        self.organ_loc_shared_memory.close()
        self.organ_loc_shared_memory.unlink()

    def request_load_image(self, patient_id, series_id, min_slice: int, max_slice: int,
                           segmentation_dataset_folder: str, data_hdf5_cropped_folder: str,
                           elastic_augmentation: bool, load_perslice_segmentation: bool):
        self.image_loading_pipe_send.send({
            "patient_id": patient_id,
            "series_id": series_id,
            "min_slice": min_slice,
            "max_slice": max_slice,
            "segmentation_dataset_folder": segmentation_dataset_folder,
            "data_hdf5_cropped_folder": data_hdf5_cropped_folder,
            "elastic_augmentation": elastic_augmentation,
            "load_perslice_segmentation": load_perslice_segmentation
        })

    def get_requested_image(self):
        self.image_required_flag.value = True
        self.image_available_lock.acquire(block=True)

        if self.has_segmentation_flag.value:
            return self.image_shared_memory_array.copy(), self.organ_loc_shared_memory_array.copy(), self.organ_segmentation_shared_memory_array.copy()
        else:
            return self.image_shared_memory_array.copy(), self.organ_loc_shared_memory_array.copy(), None


organ_sampling_depth: int = None
organ_id: int = None
target_width: int = None
target_height: int = None
loader_workers: list[SliceLoaderWorker] = None
num_loader_workers: int = None
max_angle: float = None
def initialize_async_organ_sampler(sampling_depth,
                                   o_id,
                                   organ_width,
                                   organ_height,
                                   num_workers=8, name = ""):
    global organ_sampling_depth, organ_id, target_width, target_height, max_angle
    global loader_workers, num_loader_workers
    assert sampling_depth % 2 == 1, "slice_region_width must be odd"
    organ_sampling_depth = sampling_depth
    organ_id = o_id
    loader_workers = []
    num_loader_workers = num_workers

    # Compute target width and height
    max_angle = 15 * np.pi / 180
    cur_angle = np.arctan2(organ_height, organ_width)
    diag = np.hypot(organ_height + 64.0, organ_width + 64.0)
    target_width = int(np.ceil(diag * max(np.sin(cur_angle + max_angle), np.sin(cur_angle - max_angle))))
    target_height = int(np.ceil(diag * max(np.cos(cur_angle + max_angle), np.cos(cur_angle - max_angle))))

    for k in range(num_workers):
        loader_workers.append(SliceLoaderWorker("{}_loader_{}".format(name, k),
                                                organ_sampling_depth=organ_sampling_depth,
                                                organ_id=organ_id,
                                                target_width=target_width,
                                                target_height=target_height))

def clean_and_destroy_organ_sampler():
    global loader_workers, num_loader_workers
    for k in range(num_loader_workers):
        loader_workers[k].terminate()
    del loader_workers
    del num_loader_workers

def load_image(patient_ids: list,
               series_ids: list,
               organ_id: int, organ_height: int, organ_width: int,  # organ id, expected organ height and width
               stage1_information: manager_stage1_results.Stage1ResultsManager,
               organ_sampling_depth=9,
               translate_rotate_augmentation=False,
               elastic_augmentation=False,
               load_perslice_segmentation=False,
               data_info_folder="data_hdf5_cropped") -> tuple[torch.Tensor, torch.Tensor]:
    assert len(patient_ids) == len(series_ids), "patient_ids and series_ids must have the same length"
    assert organ_width % 32 == 0, "Organ width must be divisible by 32"
    assert organ_height % 32 == 0, "Organ height must be divisible by 32"
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
    perslice_segmentation_batch = torch.zeros((batch_size, 1, organ_sampling_depth, req_rot_h, req_rot_w),
                                              dtype=torch.float32,
                                              device=config.device) if load_perslice_segmentation else None
    worker_used = 0
    for k in range(batch_size):
        organ_slice_min, organ_slice_max = stage1_information.get_organ_slicelocs(int(series_ids[k]), organ_id)
        loader_workers[worker_used % num_loader_workers].request_load_image(
            str(patient_ids[k]), str(series_ids[k]),
            organ_slice_min, organ_slice_max,
            stage1_information.segmentation_dataset_folder,
            data_info_folder,
            elastic_augmentation, load_perslice_segmentation)
        worker_used += 1

    worker_used = 0
    for k in range(batch_size):
        image, organ_location, organ_segmentation = loader_workers[worker_used % num_loader_workers].get_requested_image()

        image_batch[k, 0, ...].copy_(torch.from_numpy(image), non_blocking=True)
        organ_loc_batch[k, 0, ...].copy_(torch.from_numpy(organ_location), non_blocking=True)
        if load_perslice_segmentation:
            perslice_segmentation_batch[k, 0, ...].copy_(torch.from_numpy(organ_segmentation), non_blocking=True)

        worker_used += 1

    with torch.no_grad():
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
            if load_perslice_segmentation:
                perslice_segmentation_batch = image_sampler_augmentations.apply_displacement_field3D_simple(
                    perslice_segmentation_batch.reshape(batch_size * organ_sampling_depth, 1, req_rot_h, req_rot_w),
                    displacement_field.view(batch_size * organ_sampling_depth, req_rot_h, req_rot_w, 2)) \
                    .view(batch_size, 1, organ_sampling_depth, req_rot_h, req_rot_w)

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

            if load_perslice_segmentation:
                perslice_segmentation_batch = image_sampler_augmentations.rotate(perslice_segmentation_batch,
                                                                                 list(rotation_angles))

        ## Crop the image to the desired size, and apply translation augmentation if necessary
        final_image_batch = torch.zeros((batch_size, 1, organ_sampling_depth, organ_height, organ_width),
                                        dtype=torch.float32,
                                        device=config.device)
        final_perslice_segmentation_batch = torch.zeros(
            (batch_size, 1, organ_sampling_depth, organ_height, organ_width), dtype=torch.float32,
            device=config.device) if load_perslice_segmentation else None
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
            elif x_max > req_rot_w:
                x_min -= (x_max - req_rot_w)
                x_max = req_rot_w
            if y_min < 0:
                y_max -= y_min
                y_min = 0
            elif y_max > req_rot_h:
                y_min -= (y_max - req_rot_h)
                y_max = req_rot_h

            # apply translation augmentation
            if translate_rotate_augmentation:
                left_available, right_available = x_min, req_rot_w - x_max
                top_available, bottom_available = y_min, req_rot_h - y_max
                x_translation = np.random.randint(-min(left_available, 48), min(right_available, 48) + 1)
                y_translation = np.random.randint(-min(top_available, 48), min(bottom_available, 48) + 1)
                x_min, x_max = x_min + x_translation, x_max + x_translation
                y_min, y_max = y_min + y_translation, y_max + y_translation

            # crop the image
            final_image_batch[k, 0, ...].copy_(image_batch[k, 0, :, y_min:y_max, x_min:x_max], non_blocking=True)
            if load_perslice_segmentation:
                final_perslice_segmentation_batch[k, 0, ...].copy_(
                    perslice_segmentation_batch[k, 0, :, y_min:y_max, x_min:x_max], non_blocking=True)

    return final_image_batch, final_perslice_segmentation_batch

