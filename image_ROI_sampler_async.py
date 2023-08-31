import h5py
import os
import numpy as np
import torch
import torchvision.transforms.functional
import ctypes
import multiprocessing
import multiprocessing.shared_memory
import time
import gc

import config
import image_ROI_sampler

def image_loading_subprocess(image_loading_pipe_recv, running: multiprocessing.Value,
                             max_slice_region_width: int, image_width: int, image_height: int,
                             image_available_lock: multiprocessing.Lock,
                             image_required_flag: multiprocessing.Value,
                             worker_name: str, buffer_max_size: 5):
    try:
        print("Subprocess {} starting...".format(worker_name))

        pending_images = []
        buffered_images = []

        image_shared_memory = multiprocessing.shared_memory.SharedMemory(create=False, name="{}_image".format(worker_name))

        image_shared_memory_array = np.ndarray((max_slice_region_width, image_height, image_width), dtype=np.float32, buffer=image_shared_memory.buf)

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

                patient_id = str(load_info["patient_id"])
                series_id = str(load_info["series_id"])
                slice_locs = np.array(load_info["slices"])
                with h5py.File(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "ct_3D_image.hdf5"), "r") as f:
                    ct_3D_image = f["ct_3D_image"]
                    image_slice = ct_3D_image[slice_locs, ...]

                buffered_images.append({"image": image_slice})

            # if buffer not empty, and image required, load image from buffer
            if image_required_flag.value and (len(buffered_images) > 0):
                # set flag false
                image_required_flag.value = False

                # place data into shared memory
                image_data = buffered_images.pop(0)
                image_shared_memory_array[:image_data["image"].shape[0], ...] = image_data["image"]

                # release lock
                image_available_lock.release()

            time.sleep(0.003)

            run_time += 1
            if run_time % 10000 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("Subprocess interrupted....")

    image_shared_memory.close()
    print("Subprocess terminated.")

class SliceLoaderWorker:
    def __init__(self, worker_name: str, max_slice_region_width=9, image_width=576, image_height=512):
        self.image_width = image_width
        self.image_height = image_height
        image_loading_pipe_recv, self.image_loading_pipe_send = multiprocessing.Pipe(duplex=False)
        self.running = multiprocessing.Value(ctypes.c_bool, True)
        self.image_available_lock = multiprocessing.Lock()
        self.image_required_flag = multiprocessing.Value(ctypes.c_bool, True)
        self.image_available_lock.acquire(block=True)

        img = np.zeros((max_slice_region_width, image_height, image_width), dtype=np.float32)
        self.image_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=img.nbytes, name="{}_image".format(worker_name))
        del img
        self.image_shared_memory_array = np.ndarray((max_slice_region_width, image_height, image_width), dtype=np.float32, buffer=self.image_shared_memory.buf)

        self.process = multiprocessing.Process(target=image_loading_subprocess, args=(image_loading_pipe_recv, self.running,
                                                                                        max_slice_region_width, image_width, image_height,
                                                                                        self.image_available_lock,
                                                                                        self.image_required_flag, worker_name, 5))
        self.process.start()

    def terminate(self):
        self.running.value = False
        self.process.join()
        self.image_shared_memory.close()
        self.image_shared_memory.unlink()

    def request_load_image(self, load_info: dict):
        self.image_loading_pipe_send.send(load_info)

    def get_requested_image(self, current_slice_width: int):
        self.image_required_flag.value = True
        self.image_available_lock.acquire(block=True)

        return self.image_shared_memory_array[:current_slice_width, ...].copy()


slice_region_width: int = None
loader_workers: list[SliceLoaderWorker] = None
num_loader_workers: int = None
def initialize_async_ROI_sampler(max_slice_region_width = 9, num_workers=8, name = ""):
    global slice_region_width, loader_workers, num_loader_workers
    assert max_slice_region_width % 2 == 1, "slice_region_width must be odd"
    slice_region_width = max_slice_region_width
    loader_workers = []
    num_loader_workers = num_workers
    for k in range(num_workers):
        loader_workers.append(SliceLoaderWorker("{}loader_{}".format(name, k),
                                                max_slice_region_width=max_slice_region_width))

def load_image(patient_id: str,
               series_id: str,
               slices = 15,
               slices_random=False,
               augmentation=False) -> (torch.Tensor, torch.Tensor):
    global slice_region_width
    assert slice_region_width is not None, "You must initialize the async sampler with initialize_async_ROI_sampler()"
    slice_region_radius = (slice_region_width - 1) / 2

    # get slope and slice stride corresponding to 1cm
    slope = image_ROI_sampler.shape_info.loc[series_id, "mean_slope"]
    slope_abs = np.abs(slope)
    slice_stride = 10 / slope_abs
    if slice_stride > 1:
        slice_span = np.linspace(-int(slice_region_radius * slice_stride), int(slice_region_radius * slice_stride),
                                 slice_region_width, dtype=np.int32)
        slice_span[slice_region_radius] = 0
        contracted = False
        loaded_temp_depth = slice_region_width
    else:
        slice_span = np.arange(-int(slice_region_radius * slice_stride), int(slice_region_radius * slice_stride) + 1, dtype=np.int32)
        contracted = True
        loaded_temp_depth = len(slice_span)

    with torch.no_grad():
        with h5py.File(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "ct_3D_image.hdf5"), "r") as f:
            ct_3D_image = f["ct_3D_image"]
            total_slices = ct_3D_image.shape[0]
            original_height, original_width = ct_3D_image.shape[1], ct_3D_image.shape[2]
            max_angle = image_ROI_sampler.compute_max_angle(original_height, original_width)
            # at most 15 degrees
            maxdev = min(max_angle, np.pi / 12)
            if maxdev > 0.00872664626:  # if max deviation <= 0.5 degrees, we don't rotate
                angle = np.random.uniform(-maxdev, maxdev)
            else:
                angle = 0.0

        # randomly pick slices, region of interest
        slice_poses = np.linspace(0, total_slices - 1, slices + 2, dtype=np.int32)[1:-1] # equidistant
        if slices_random:
            dist = (np.min(np.diff(slice_poses)) // 2) - 1
            if dist > 1:
                slice_poses = slice_poses + np.random.randint(-dist, dist + 1, size=slices)
        slice_poses = np.clip(np.sort(slice_poses), -slice_span[0], total_slices - 1 - slice_span[-1])

        # sample the images and the segmentation now
        image = torch.zeros((slices, 1, loaded_temp_depth, original_height, original_width), dtype=torch.float32, device=config.device)
        worker_assign = 0
        for k in range(slices):
            slice_pos = slice_poses[k]
            worker_used = (worker_assign % num_loader_workers)
            loader_workers[worker_used].request_load_image({"patient_id": patient_id,
                                                            "series_id": series_id,
                                                            "slices": list(slice_pos + slice_span)})
            worker_assign += 1
        with h5py.File(os.path.join("data_segmentation_hdf_cropped", str(series_id) + ".hdf5"), "r") as f:
            segmentation_3D_image = f["segmentation_arr"]
            segmentation_raw = segmentation_3D_image[slice_poses, ...].astype(dtype=bool)
            segmentations = np.zeros((slices, original_height, original_width, 4), dtype=bool)
            segmentations[..., :2] = segmentation_raw[..., :2]
            segmentations[..., 2] = np.any(segmentation_raw[..., 2:4], axis=-1)
            segmentations[..., 3] = segmentation_raw[..., 4]
            del segmentation_raw
        worker_assign = 0
        for k in range(slices):
            worker_used = (worker_assign % num_loader_workers)
            image_slice = loader_workers[worker_used].get_requested_image(loaded_temp_depth)
            image[k, 0, ...].copy_(torch.from_numpy(image_slice), non_blocking=True)
            worker_assign += 1

        segmentations = torch.tensor(segmentations, dtype=torch.float32, device=config.device).permute((0, 3, 1, 2))

        # reshape the depth dimension if contracted
        if contracted:
            image = torchvision.transforms.functional.interpolate(image,
                                size=(slice_region_width, original_height, original_width), mode="trilinear")

        assert image.shape[-2] == segmentations.shape[-2] and image.shape[-1] == segmentations.shape[-1]
        # whether augmentation or not, we return a (slices, C, slice_depth, 512, 576) image
        if augmentation:
            # rotate
            image = torchvision.transforms.functional.rotate(image, angle * 180 / np.pi, expand=True, fill=0.0,
                                                             interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            segmentations = torchvision.transforms.functional.rotate(segmentations, angle * 180 / np.pi, expand=True, fill=0.0,
                                                             interpolation=torchvision.transforms.InterpolationMode.NEAREST)

            # expand randomly
            assert image.shape[-2] <= 512 and image.shape[-1] <= 576
            assert segmentations.shape[-2] <= 512 and segmentations.shape[-1] <= 576
            assert image.shape[-2] == segmentations.shape[-2] and image.shape[-1] == segmentations.shape[-1]
            height_required = 512 - image.shape[-2]
            width_required = 576 - image.shape[-1]

            top = np.random.randint(0, height_required + 1)
            bottom = height_required - top
            left = np.random.randint(0, width_required + 1)
            right = width_required - left

            image = torch.nn.functional.pad(image, (left, right, top, bottom))
            segmentations = torch.nn.functional.pad(segmentations, (left, right, top, bottom))
        else:
            top = (512 - image.shape[-2]) // 2
            bottom = 512 - image.shape[-2] - top
            left = (576 - image.shape[-1]) // 2
            right = 576 - image.shape[-1] - left

            image = torch.nn.functional.pad(image, (left, right, top, bottom))
            segmentations = torch.nn.functional.pad(segmentations, (left, right, top, bottom))

    # downscale segmentations by 32 with max pooling
    segmentations = torch.nn.functional.max_pool2d(segmentations, kernel_size=32, stride=32)
    return image, segmentations
