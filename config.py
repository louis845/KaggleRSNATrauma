import argparse
import torch

device = None

def add_argparse_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--device", type=int, default=0, help="Which GPU device to use")

def parse_args(args: argparse.Namespace):
    global device
    device_str = "cuda:{}".format(args.device)
    device = torch.device(device_str)

    print("Using device: {}".format(torch.cuda.get_device_properties(device)))

