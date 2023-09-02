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
import image_sampler_augmentations

def image_loading_subprocess(image_loading_pipe_recv, running: multiprocessing.Value,
                             max_slice_region_depth: int, max_image_width: int, max_image_height: int,
                             loaded_image_depth: multiprocessing.Value, loaded_image_width: multiprocessing.Value, loaded_image_height: multiprocessing.Value,
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
                slice_locs = np.array(load_info["slices"])
                slice_center = load_info["slice_center"]
                elastic_augmentation = load_info["elastic_augmentation"]

                # load image and apply augmentation if needed
                with h5py.File(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "ct_3D_image.hdf5"), "r") as f:
                    ct_3D_image = f["ct_3D_image"]
                    image_slice = ct_3D_image[slice_locs, ...]
                image_slice = torch.from_numpy(image_slice)
                original_height = image_slice.shape[1]
                original_width = image_slice.shape[2]

                with h5py.File(os.path.join("data_segmentation_hdf_cropped", str(series_id) + ".hdf5"), "r") as f:
                    segmentation_3D_image = f["segmentation_arr"]
                    if not use_3d:
                        segmentation_raw = segmentation_3D_image[slice_center, ...].astype(dtype=bool)
                        segmentation_slice = np.zeros((original_height, original_width, 4), dtype=bool)
                        segmentation_slice[..., :2] = segmentation_raw[..., :2]
                        segmentation_slice[..., 2] = np.any(segmentation_raw[..., 2:4], axis=-1)
                        segmentation_slice[..., 3] = segmentation_raw[..., 4]
                    else:
                        segmentation_raw = segmentation_3D_image[slice_locs, ...].astype(dtype=bool)
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

                # apply elastic deformation to height width
                if elastic_augmentation:
                    displacement_field = image_sampler_augmentations.generate_displacement_field(original_width,
                                                                                                 original_height)
                    displacement_field = torch.from_numpy(displacement_field)
                    image_slice = torchvision.transforms.functional.elastic_transform(
                        image_slice.unsqueeze(1), displacement_field,
                        interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(1)
                    if not use_3d:
                        segmentation_slice = torchvision.transforms.functional.elastic_transform(
                            segmentation_slice.unsqueeze(1), displacement_field,
                            interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(1)
                    else:
                        assert segmentation_slice.shape[0] == 4 and segmentation_slice.shape[1] == len(slice_locs)
                        segmentation_slice = torchvision.transforms.functional.elastic_transform(
                            segmentation_slice.reshape(4 * len(slice_locs), 1, original_height, original_width),
                            displacement_field,
                            interpolation=torchvision.transforms.InterpolationMode.NEAREST).view(4, len(slice_locs),
                                                                                                 original_height,
                                                                                                 original_width)

                buffered_images.append({"image": image_slice, "segmentation": segmentation_slice})

            # if buffer not empty, and image required, load image from buffer
            if image_required_flag.value and (len(buffered_images) > 0):
                # set flag false
                image_required_flag.value = False

                # place data into shared memory
                image_data = buffered_images.pop(0)
                img_d, img_h, img_w = image_data["image"].shape
                image_shared_memory_array[:img_d, :img_h, :img_w] = image_data["image"]
                seg_shared_memory_array[:, :img_d, :img_h, :img_w] = image_data["segmentation"]

                # set value to pass to main process
                loaded_image_depth.value = img_d
                loaded_image_height.value = img_h
                loaded_image_width.value = img_w

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

        self.process = multiprocessing.Process(target=image_loading_subprocess, args=(image_loading_pipe_recv, self.running,
                                                                                        max_slice_region_depth, max_image_width, max_image_height,
                                                                                        self.loaded_image_depth, self.loaded_image_width, self.loaded_image_height,
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

    def request_load_image(self, patient_id, series_id, slices: list[int], slice_center: int, elastic_augmentation: bool):
        self.image_loading_pipe_send.send({
            "patient_id": patient_id,
            "series_id": series_id,
            "slices": slices,
            "slice_center": slice_center,
            "elastic_augmentation": elastic_augmentation
        })

    def get_requested_image(self):
        self.image_required_flag.value = True
        self.image_available_lock.acquire(block=True)

        return self.image_shared_memory_array[:self.loaded_image_depth.value, :self.loaded_image_height.value, :self.loaded_image_width].copy(),\
                self.seg_shared_memory_array[:, :self.loaded_image_depth.value, :self.loaded_image_height.value, :self.loaded_image_width].copy()


slice_region_depth: int = None
loader_workers: list[SliceLoaderWorker] = None
num_loader_workers: int = None
def initialize_async_ROI_sampler(max_slice_region_depth = 9, use_3d=False, num_workers=8, name = ""):
    global slice_region_depth, loader_workers, num_loader_workers
    assert max_slice_region_depth % 2 == 1, "slice_region_width must be odd"
    slice_region_depth = max_slice_region_depth
    loader_workers = []
    num_loader_workers = num_workers
    for k in range(num_workers):
        loader_workers.append(SliceLoaderWorker("{}_loader_{}".format(name, k),
                                                max_slice_region_depth=max_slice_region_depth, use_3d=use_3d))

def clean_and_destroy_ROI_sampler():
    global slice_region_depth, loader_workers, num_loader_workers
    for k in range(num_loader_workers):
        loader_workers[k].terminate()
    del loader_workers
    del slice_region_depth
    del num_loader_workers

def load_image(patient_id: str,
               series_id: str,
               slices = 15,
               segmentation_region_depth = 1,
               slices_random=False,
               translate_rotate_augmentation=False,
               elastic_augmentation=False) -> (torch.Tensor, torch.Tensor):
    global slice_region_depth, loader_workers, num_loader_workers
    assert segmentation_region_depth % 2 == 1, "segmentation_region_width must be odd"
    assert segmentation_region_depth <= slice_region_depth, "segmentation_region_width must be less than or equal to slice_region_width"
    slice_region_radius = (slice_region_depth - 1) // 2

    # get slope and slice stride corresponding to 0.5cm
    slope = image_ROI_sampler.shape_info.loc[int(series_id), "mean_slope"]
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
            max_angle = image_ROI_sampler.compute_max_angle(original_height, original_width)
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
        if segmentation_region_depth == 1:
            segmentations = torch.zeros((slices, 4, original_height, original_width), dtype=torch.float32, device=config.device)
        else:
            segmentations = torch.zeros((slices, 4, loaded_temp_depth, original_height, original_width), dtype=torch.float32, device=config.device)

        worker_used = 0
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

            loader_workers[worker_used % num_loader_workers].request_load_image(patient_id, series_id, list(cur_slice_depths), slice_pos, elastic_augmentation=elastic_augmentation)
            worker_used += 1

        worker_used = 0
        for k in range(slices):
            image_slice, segmentation_slice = loader_workers[worker_used % num_loader_workers].get_requested_image()
            image[k, 0, ...].copy_(image_slice, non_blocking=True)
            segmentations[k, ...].copy_(segmentation_slice, non_blocking=True)
            worker_used += 1

        # reshape the depth dimension if contracted
        if contracted:
            image = torch.nn.functional.interpolate(image,
                                size=(slice_region_depth, original_height, original_width), mode="trilinear")
            if segmentation_region_depth > 1:
                segmentations = (torch.nn.functional.interpolate(segmentations,
                                size=(slice_region_depth, original_height, original_width), mode="trilinear") > 0.5).to(torch.float32)
        if segmentation_region_depth > 1:
            segmentations = segmentations[:, :, slice_region_radius - (segmentation_region_depth - 1) // 2:slice_region_radius + (segmentation_region_depth + 1) // 2, ...]

        assert image.shape[-2] == segmentations.shape[-2] and image.shape[-1] == segmentations.shape[-1]
        # whether augmentation or not, we return a (slices, C, slice_depth, 512, 576) image
        if translate_rotate_augmentation:
            # rotate
            image = torchvision.transforms.functional.rotate(image.squeeze(1), angle * 180 / np.pi, expand=True,
                                                             interpolation=torchvision.transforms.InterpolationMode.NEAREST).unsqueeze(1)
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
        if segmentation_region_depth == 1:
            segmentations = torch.nn.functional.max_pool2d(segmentations, kernel_size=32, stride=32)
        else:
            segmentations = torch.nn.functional.max_pool2d(segmentations.view(slices, 4 * segmentation_region_depth, 512, 576),
                                                kernel_size=32, stride=32).view(slices, 4, segmentation_region_depth, 16, 18)
    return image, segmentations
