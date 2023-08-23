import os
import tqdm
import argparse
import pandas as pd
import hashlib
import datetime

ground_truth_hash = pd.read_csv("hdf5_hash.csv")
hash_dict = {}
for k in range(len(ground_truth_hash)):
    patient_id = ground_truth_hash["patient_id"][k]
    series_id = ground_truth_hash["series_id"][k]
    sha1sum = ground_truth_hash["sha1sum"][k]
    hash_dict[(patient_id, series_id)] = sha1sum

problem_info = {"patient_id": [], "series_id": [], "problem": []}
for patient_id in tqdm.tqdm(os.listdir("data_hdf5")):
    patient_folder = os.path.join("data_hdf5", patient_id)
    if not os.path.isdir(patient_folder):
        continue
    for series_id in os.listdir(patient_folder):
        series_folder = os.path.join(patient_folder, series_id)

        ct_file = os.path.join(series_folder, "ct_3D_image.hdf5")

        # compute sha1sum
        sha1sum = hashlib.sha1(open(ct_file, "rb").read()).hexdigest()
        
        problem_info["patient_id"].append(patient_id)
        problem_info["series_id"].append(series_id)
        problem_info["problem"].append(hash_dict[(patient_id, series_id)] != sha1sum)

if not os.path.isdir("integrity_check"):
    os.makedirs("integrity_check")

problem_info = pd.DataFrame(problem_info)
# save to integrity_check/<current_time>.csv
problem_info.to_csv(os.path.join("integrity_check", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"), index=False)
