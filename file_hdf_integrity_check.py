import os
import tqdm
import argparse
import pandas as pd
import hashlib
import datetime
import h5py
import traceback

problem_info = {"patient_id": [], "series_id": [], "problem": []}
for patient_id in tqdm.tqdm(os.listdir("data_hdf5")):
    patient_folder = os.path.join("data_hdf5", patient_id)
    if not os.path.isdir(patient_folder):
        continue
    for series_id in os.listdir(patient_folder):
        series_folder = os.path.join(patient_folder, series_id)

        with h5py.File(os.path.join(series_folder, "ct_3D_image.hdf5"), "r") as f:
            try:
                f["ct_3D_image"][()]
            except OSError as e:
                print("---------------------- Error in {} {} ----------------------".format(patient_id, series_id))
                traceback.print_exc()
                problem_info["patient_id"].append(patient_id)
                problem_info["series_id"].append(series_id)
                problem_info["problem"].append(True)
                continue


        problem_info["patient_id"].append(patient_id)
        problem_info["series_id"].append(series_id)
        problem_info["problem"].append(False)

if not os.path.isdir("integrity_check"):
    os.makedirs("integrity_check")

problem_info = pd.DataFrame(problem_info)
# save to integrity_check/<current_time>.csv
problem_info.to_csv(os.path.join("integrity_check", "hdf_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"),
                    index=False)

# if there is any problem, exit with code -1
if problem_info["problem"].any():
    print("Not all ok!")
    exit(-1)
print("All ok.")
exit(0)
