import os
import argparse

model_path = "models"
if not os.path.exists(model_path):
    os.mkdir(model_path)

def add_argparse_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, required=True, help="Name of the new model")
    parser.add_argument("--load_model", type=str, default=None, help="Previous model checkpoint to load from")

def parse_args(args: argparse.Namespace):
    new_model_path = os.path.join(model_path, args.model)
    assert not os.path.exists(new_model_path), "Model {} already exists".format(args.model)
    if args.load_model is not None:
        previous_model_path = os.path.join(model_path, args.load_model)
        assert os.path.exists(previous_model_path), "Previous model checkpoint does not exist"
    else:
        previous_model_path = None
    os.mkdir(new_model_path)
    return new_model_path, previous_model_path
