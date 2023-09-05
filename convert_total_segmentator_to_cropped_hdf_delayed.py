import time
import os
import argparse

import convert_total_segmentator_to_cropped_hdf
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crops the results from Total Segmentator.")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--partition", type=int, default=0, help="Partition")
    args = parser.parse_args()
    stride = args.stride
    partition = args.partition

    patient_injuries = pd.read_csv("data/train.csv", index_col=0)
    patients_with_injuries = list(patient_injuries.loc[patient_injuries["any_injury"] == 1].index)

    series_with_expert_segmentations = [int(x[:-4]) for x in os.listdir("data/segmentations")]
    count = 0
    for patient_id in patients_with_injuries:
        patient_folder = os.path.join("data", "train_images", str(patient_id))
        for series_id in os.listdir(patient_folder):
            if int(series_id) not in series_with_expert_segmentations:
                count += 1

    print("CT segmentations required from Total Segmentator:   {}".format(count))

    while True:
        completed = len(os.listdir("total_segmentator_results"))
        if completed == count:
            print("Total segmentator process completed. Cropping and converting...")
            break
        else:
            print("Total segmentator processes still running. Completed {} out of {}".format(completed, count))
        time.sleep(600)

    time.sleep(100)
    
    convert_total_segmentator_to_cropped_hdf.main_func(stride, partition)