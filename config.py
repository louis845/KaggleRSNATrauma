import argparse
import torch

device = None

def add_argparse_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--device", type=int, default=0, help="Which GPU device to use")
    parser.add_argument("--memory_limit", type=float, default=None, help="Limit the GPU memory to this fraction (0-1) of the total memory")

def parse_args(args: argparse.Namespace):
    global device
    device_str = "cuda:{}".format(args.device)
    device = torch.device(device_str)

    print("Using device: {}".format(torch.cuda.get_device_properties(device)))

    if args.memory_limit is not None:
        assert args.memory_limit > 0 and args.memory_limit <= 1, "Memory limit must be between 0 and 1"
        torch.cuda.set_per_process_memory_fraction(args.memory_limit, device=device)
        print("Limiting GPU memory ratio: {}".format(args.memory_limit))
    else:
        print("Not limiting GPU memory.")

