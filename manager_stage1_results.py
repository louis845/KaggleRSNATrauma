import os

import pandas as pd
import numpy as np
import h5py
import tqdm

SEGMENTATION_RESULTS_FOLDER = "stage1_organ_segmentator"
SEGMENTATION_EVAL_FOLDER = "stage1_organ_segmentator_eval"

class Stage1ResultsManager:
    organs = ["liver", "spleen", "kidney", "bowel"]

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.segmentation_dataset_folder = os.path.join(SEGMENTATION_RESULTS_FOLDER, dataset_name)
        self.segmentation_eval_folder = os.path.join(SEGMENTATION_EVAL_FOLDER, dataset_name)

        assert os.path.isdir(self.segmentation_dataset_folder), "Segmentations not available for " + dataset_name
        assert os.path.isdir(self.segmentation_eval_folder), "Segmentation preds not available for " + dataset_name

        self.series = [int(filename[:-4]) for filename in os.listdir(self.segmentation_dataset_folder)
                                if filename.endswith(".csv")]
        self.seg_info = pd.read_csv(os.path.join(self.segmentation_eval_folder, "entrywise_preds.csv"), index_col=0)
        assert set(self.series) == set(self.seg_info.index), "Series mismatch"

    def is_series_good(self, series_id: int):
        return (self.seg_info.loc[series_id, "slice_recall_liver"] > 0.9) and \
                (self.seg_info.loc[series_id, "slice_recall_spleen"] > 0.9) and \
                (self.seg_info.loc[series_id, "slice_recall_kidney"] > 0.9)

    def validate_patient_ids_contained(self, patient_ids: list):
        patient_ids = [str(x) for x in patient_ids]
        for patient_id in patient_ids:
            for series_id in os.listdir(os.path.join("data", "train_images", patient_id)):
                assert int(series_id) in self.series, "Some patients contain series not in this stage1 result!"

    def list_good_series(self, patient_id):
        good_series = []
        for series_id in os.listdir(os.path.join("data", "train_images", str(patient_id))):
            assert int(series_id) in self.series
            if self.is_series_good(int(series_id)):
                good_series.append(series_id)
        return good_series

    def restrict_patient_ids_to_good_series(self, patient_ids: list):
        good_patient_ids = []
        for patient_id in patient_ids:
            if len(self.list_good_series(patient_id)) > 0:
                good_patient_ids.append(patient_id)
        return good_patient_ids

    def restrict_patient_ids_to_organs(self, patient_ids: list, organ_id: int):
        good_patient_ids = []
        for patient_id in patient_ids:
            for series_id in os.listdir(os.path.join("data", "train_images", str(patient_id))):
                if self.has_organ(int(series_id), organ_id):
                    good_patient_ids.append(patient_id)
                    break
        return good_patient_ids

    def get_min2d_size(self, organ_id: int):
        """
        Get the minimum enclosing 2D size (height x width) for the given organ
        Organs:
        0 - liver
        1 - spleen
        2 - kidney
        3 - bowel
        """
        organ_size_file = os.path.join(self.segmentation_eval_folder, "min2d_size_{}.txt".format(self.organs[organ_id]))
        if os.path.isfile(organ_size_file):
            with open(organ_size_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2
                assert lines[0].strip() == "Organ {} (H x W):".format(self.organs[organ_id])
                height, width = lines[1].strip().split(" x ")
                enclosing_height = int(height)
                enclosing_width = int(width)
        else:
            enclosing_height = 0
            enclosing_width = 0
            for series_id in tqdm.tqdm(self.series):
                if self.is_series_good(series_id):
                    with h5py.File(os.path.join(self.segmentation_dataset_folder, str(series_id) + ".hdf5"), "r") as f:
                        organ_mask = f["organ_location"][organ_id, ...] > 0 # shape: (H, W)

                    if np.any(organ_mask):
                        heights = np.any(organ_mask, axis=-1)
                        widths = np.any(organ_mask, axis=-2)

                        heights = np.argwhere(heights)
                        widths = np.argwhere(widths)

                        height = heights.max() + 1 - heights.min()
                        width = widths.max() + 1 - widths.min()

                        enclosing_height = max(enclosing_height, height)
                        enclosing_width = max(enclosing_width, width)

            # write to file
            with open(organ_size_file, "w") as f:
                f.writelines([
                    "Organ {} (H x W):\n".format(self.organs[organ_id]),
                    "{} x {}\n".format(enclosing_height, enclosing_width)
                ])

        return enclosing_height, enclosing_width

    def has_organ(self, series_id: int, organ_id: int) -> bool:
        assert series_id in self.series

        organ_info = pd.read_csv(os.path.join(self.segmentation_dataset_folder, str(series_id) + ".csv"), index_col=0)
        return bool(organ_info.loc[self.organs[organ_id], "found"])

    def get_organ_slicelocs(self, series_id: int, organ_id: int) -> tuple[int, int]:
        assert series_id in self.series

        organ_info = pd.read_csv(os.path.join(self.segmentation_dataset_folder, str(series_id) + ".csv"), index_col=0)
        return organ_info.loc[self.organs[organ_id], "left"], organ_info.loc[self.organs[organ_id], "right"]

    def get_dual_series(self, patient_ids: list[int], organ_id: int):
        s1 = []
        s2 = []
        for patient_id in patient_ids:
            series = os.listdir(os.path.join("data", "train_images", str(patient_id)))
            series = [int(x) for x in series if self.has_organ(int(x), organ_id) and self.is_series_good(int(x))]
            if len(series) == 1:
                s1.append(series[0])
                s2.append(series[0])
            elif len(series) == 2:
                if np.random.rand() > 0.5:
                    s1.append(series[0])
                    s2.append(series[1])
                else:
                    s1.append(series[1])
                    s2.append(series[0])
        return s1, s2

if __name__ == "__main__":
    datasets = [
        "ROI_classifier_fold0_train",
        "ROI_classifier_fold0_val",
        "ROI_classifier_fold1_train",
        "ROI_classifier_fold1_val",
        "ROI_classifier_fold2_train",
        "ROI_classifier_fold2_val"
    ]

    for dataset in datasets:
        mgr = Stage1ResultsManager(dataset)
        organs = ["liver", "spleen", "kidney", "bowel"]
        for i, organ in enumerate(organs):
            print("Min 2D size in {} for {}: {}".format(dataset, organ, mgr.get_min2d_size(i)))
