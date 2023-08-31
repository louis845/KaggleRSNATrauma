import pandas as pd
import os

def get_patients_with_expert_segmentation() -> list:
    series_segmentation = [int(segmentation[:-4]) for segmentation in os.listdir("data/segmentations")]
    meta_by_series = pd.read_csv("data/train_series_meta.csv", index_col=1)
    patients_with_segmentation = list(set(list(meta_by_series.loc[series_segmentation]["patient_id"])))
    meta_patients_with_segs = meta_by_series.loc[meta_by_series["patient_id"].isin(patients_with_segmentation)]
    for k in range(len(meta_patients_with_segs)):
        assert int(meta_patients_with_segs.index[
                       k]) in series_segmentation, "Some patient has > 1 series, but not all of them have segmentations."

    return patients_with_segmentation

