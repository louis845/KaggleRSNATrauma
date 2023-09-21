import os
import argparse

import manager_stage1_results

if __name__ == "__main__":
    # Extracts the images and labels from the stage 1 results to be placed into a folder for training with fastai
    parser = argparse.ArgumentParser(description='Extracts the images and labels from the stage 1 results to be placed into a folder for training with fastai')
    