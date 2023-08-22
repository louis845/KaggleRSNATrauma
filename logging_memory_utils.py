import torch
import time
import os

class CudaMemoryLogger:
    def __init__(self, folder, activated):
        # open file in folder/cuda_mem_log.txt
        if activated:
            self.file = open(os.path.join(folder, "cuda_mem_log.txt"), "w")
            self.closed = False
        else:
            self.closed = True

    def log(self, msg):
        if not self.closed:
            self.file.write("========================================\n")
            self.file.write("Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
            self.file.write(msg + "\n")
            self.file.write(torch.cuda.memory_summary())
            self.file.flush()

    def close(self):
        if not self.closed:
            self.file.flush()
            self.file.close()
            self.closed = True

    def __del__(self):
        self.close()

def obtain_memory_logger(folder, activated=True):
    assert os.path.isdir(folder), "Folder must be an existing directory"
    return CudaMemoryLogger(folder, activated)
