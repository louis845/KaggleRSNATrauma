import numpy as np
import torch

import os
import gc
import ctypes
import multiprocessing
import multiprocessing.shared_memory
import time

import image_sampler
import config

def image_loading_subprocess(image_loading_pipe_recv, running: multiprocessing.Value,
                             num_slices: int, image_width: int, image_height: int,
                             image_available_lock: multiprocessing.Lock,
                             image_required_flag: multiprocessing.Value,
                             worker_name: str, buffer_max_size: 5):
    try:
        print("Subprocess {} starting...".format(worker_name))

        pending_images = []
        buffered_images = []

        image_shared_memory = multiprocessing.shared_memory.SharedMemory(create=False, name="{}_image".format(worker_name))

        image_shared_memory_array = np.ndarray((num_slices, image_height, image_width), dtype=np.float32, buffer=image_shared_memory.buf)

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
                img_data = image_sampler.load_image(load_info["patient_id"], load_info["series_id"], num_slices,
                                                    load_info["slices_random"], load_info["augmentation"])
                buffered_images.append({"image": img_data})

            # if buffer not empty, and image required, load image from buffer
            if image_required_flag.value and (len(buffered_images) > 0):
                # set flag false
                image_required_flag.value = False

                # place data into shared memory
                image_data = buffered_images.pop(0)
                image_shared_memory_array[:] = image_data["image"]

                # release lock
                image_available_lock.release()

            time.sleep(0.001)

            run_time += 1
            if run_time % 10000 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("Subprocess interrupted....")

    image_shared_memory.close()
    print("Subprocess terminated.")

class ImageLoaderWorker:
    def __init__(self, worker_name: str, num_slices=15, image_width=640, image_height=640):
        self.image_width = image_width
        self.image_height = image_height
        image_loading_pipe_recv, self.image_loading_pipe_send = multiprocessing.Pipe(duplex=False)
        self.running = multiprocessing.Value(ctypes.c_bool, True)
        self.image_available_lock = multiprocessing.Lock()
        self.image_required_flag = multiprocessing.Value(ctypes.c_bool, True)
        self.image_available_lock.acquire(block=True)

        img = np.zeros((num_slices, image_height, image_width), dtype=np.float32)
        self.image_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=img.nbytes, name="{}_image".format(worker_name))
        del img
        self.image_shared_memory_array = np.ndarray((num_slices, image_height, image_width), dtype=np.float32, buffer=self.image_shared_memory.buf)

        self.process = multiprocessing.Process(target=image_loading_subprocess, args=(image_loading_pipe_recv, self.running,
                                                                                        num_slices, image_width, image_height,
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

    def get_requested_image(self):
        self.image_required_flag.value = True
        self.image_available_lock.acquire(block=True)

        return self.image_shared_memory_array.copy()


class ImageSamplerAsync:
    """
    This class is responsible for sampling images from the dataset, as a batch of images in a torch tensor.
    """
    def __init__(self, max_batch_size, num_slices=15, image_width=640, image_height=640, device: torch.device=config.device):
        self.closed = False
        self.device = device
        self.max_batch_size = max_batch_size
        self.num_slices = num_slices
        self.image_width = image_width
        self.image_height = image_height

        # buffers for the data loader.
        self.img_data_batch = torch.zeros(max_batch_size, 1, self.num_slices, self.image_height, self.image_width, device=self.device)

        self.workers = []
        for k in range(max_batch_size):
            self.workers.append(ImageLoaderWorker("worker_{}".format(k), image_width=image_width, image_height=image_height, num_slices=num_slices))

    def get_data_from_worker(self, worker_idx: int):
        img_data = self.workers[worker_idx].get_requested_image()
        assert img_data.shape == (self.num_slices, self.image_height, self.image_width)
        assert img_data.dtype == np.float32

        return img_data

    def obtain_sample_batch(self, patient_ids: list[str], series_ids: list[str], slices_random: bool, augmentation: bool):
        assert len(patient_ids) == len(series_ids)
        batch_size = len(patient_ids)

        for i in range(batch_size):
            self.workers[i].request_load_image({
                "patient_id": patient_ids[i],
                "series_id": series_ids[i],
                "slices_random": slices_random,
                "augmentation": augmentation
            })

        for i in range(batch_size):
            img_data = self.get_data_from_worker(i)

            self.img_data_batch[i, 0, ...].copy_(torch.from_numpy(img_data), non_blocking=True)

        return self.img_data_batch[:batch_size, ...]

    def close(self):
        if not self.closed:
            for k in range(self.max_batch_size):
                self.workers[k].terminate()
            self.closed = True
