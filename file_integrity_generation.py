import os
import tqdm
import argparse
import pandas as pd
import hashlib

hash_info = {"patient_id": [], "series_id": [], "sha1sum": []}
for patient_id in tqdm.tqdm(os.listdir("data_hdf5")):
    patient_folder = os.path.join("data_hdf5", patient_id)
    if not os.path.isdir(patient_folder):
        continue
    for series_id in os.listdir(patient_folder):
        series_folder = os.path.join(patient_folder, series_id)

        ct_file = os.path.join(series_folder, "ct_3D_image.hdf5")

        # compute sha1sum
        sha1sum = hashlib.sha1(open(ct_file, "rb").read()).hexdigest()

        hash_info["patient_id"].append(patient_id)
        hash_info["series_id"].append(series_id)
        hash_info["sha1sum"].append(sha1sum)

# save the dataframe
hash_info = pd.DataFrame(hash_info)
hash_info.to_csv("hdf5_hash.csv", index=False)
