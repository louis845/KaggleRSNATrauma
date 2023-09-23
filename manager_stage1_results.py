import os
import shutil
import typing

import pandas as pd
import numpy as np
import h5py
import tqdm

SEGMENTATION_RESULTS_FOLDER = "stage1_organ_segmentator"
SEGMENTATION_EVAL_FOLDER = "stage1_organ_segmentator_eval"

class Stage1ResultsManager:
    organs = ["liver", "spleen", "kidney", "bowel"]

    def __init__(self, dataset_name: str, SEGMENTATION_RESULTS_FOLDER_OVERRIDE=None):
        main_results_folder = SEGMENTATION_RESULTS_FOLDER
        if SEGMENTATION_RESULTS_FOLDER_OVERRIDE is not None:
            main_results_folder = SEGMENTATION_RESULTS_FOLDER_OVERRIDE
        self.dataset_name = dataset_name
        self.segmentation_dataset_folder = os.path.join(main_results_folder, dataset_name)
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
        assert os.path.isdir(os.path.join("data", "train_images", str(patient_id))), "Patient not found"
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

    def create_copy(self, dest_results_foler, dest_eval_folder,
                    organ_location_copy_map: typing.Callable[[str, int, pd.DataFrame, str], None]):
        # the organ_location_copy_map function is called for each series_id in self.series
        # it should handle copying of the series_id.csv file and reformat it to transformed organ locations
        # it is called via organ_location_copy_map(dataset_name: str, series_id: int, organs_info: pd.DataFrame, dest_folder: str)
        dest_segmentation_dataset_folder = os.path.join(dest_results_foler, self.dataset_name)
        dest_segmentation_eval_folder = os.path.join(dest_eval_folder, self.dataset_name)
        if not os.path.isdir(dest_segmentation_dataset_folder):
            os.makedirs(dest_segmentation_dataset_folder)

        # copy all files of self.segmentation_dataset_folder ending with .hdf5 to dest_segmentation_dataset_folder
        for filename in os.listdir(self.segmentation_dataset_folder):
            if filename.endswith(".hdf5"):
                shutil.copyfile(os.path.join(self.segmentation_dataset_folder, filename),
                                os.path.join(dest_segmentation_dataset_folder, filename))

        for series_id in tqdm.tqdm(self.series):
            organ_info = pd.read_csv(os.path.join(self.segmentation_dataset_folder, str(series_id) + ".csv"), index_col=0)
            organ_location_copy_map(self.dataset_name, series_id, organ_info, dest_segmentation_dataset_folder)

        # copy the contents of self.segmentation_eval_folder to dest_segmentation_eval_folder
        shutil.copytree(self.segmentation_eval_folder, dest_segmentation_eval_folder)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    def get_wh_area(mgr: Stage1ResultsManager, series_id: int, organ_id: int):
        with h5py.File(os.path.join(mgr.segmentation_dataset_folder, str(series_id) + ".hdf5"), "r") as f:
            organ_mask = f["organ_location"][organ_id, ...] > 0  # shape: (H, W)

        if np.any(organ_mask):
            heights = np.any(organ_mask, axis=-1)
            widths = np.any(organ_mask, axis=-2)

            heights = np.argwhere(heights)
            widths = np.argwhere(widths)

            height = heights.max() + 1 - heights.min()
            width = widths.max() + 1 - widths.min()
            area = height * width
        else:
            area = 0
            height = 0
            width = 0
        return area, height, width


    datasets = [
        "ROI_classifier_fold0_train",
        "ROI_classifier_fold0_val",
        "ROI_classifier_fold1_train",
        "ROI_classifier_fold1_val",
        "ROI_classifier_fold2_train",
        "ROI_classifier_fold2_val"
    ]

    organs = ["liver", "spleen", "kidney", "bowel"]
    per_organ_outliers = {organ: [] for organ in organs}
    for dataset in datasets:
        mgr = Stage1ResultsManager(dataset)
        for i, organ in enumerate(organs):
            print("Min 2D size in {} for {}: {}".format(dataset, organ, mgr.get_min2d_size(i)))

            info = {"series_id": [], "area": [], "height": [], "width": []}
            for series_id in tqdm.tqdm(mgr.series):
                if mgr.is_series_good(series_id):
                    area, height, width = get_wh_area(mgr, series_id, i)
                    info["series_id"].append(series_id)
                    info["area"].append(area)
                    info["height"].append(height)
                    info["width"].append(width)
            info = pd.DataFrame(info, columns=["area", "height", "width"], index=info["series_id"])

            # Find outliers. We use (height, width) as the feature vector for each image, and use the
            # interquartile range to find outliers. Outliers are defined as points that are not between
            # the 5% to 95% quantiles, in at least one dimension (H or W).
            cols = ["height", "width"]
            for col in cols:
                q1 = info[col].quantile(0.05)
                q3 = info[col].quantile(0.95)
                iqr = q3 - q1
                outliers = info[(info[col] < q1 - 1.5 * iqr) | (info[col] > q3 + 1.5 * iqr)]
                outliers = list(outliers.index)
                for outlier in outliers:
                    if outlier not in per_organ_outliers[organ]:
                        per_organ_outliers[organ].append(outlier)
                # add a column "outlier" to the info dataframe, indicating whether each image is an outlier
                info["outlier"] = info.index.isin(outliers)


            # plot 4 matplotlib plots. The first three are boxplots for the distribution of area, height, and width.
            # The last plot is a scatter plot of height vs width. Each plot should include a title, and there should
            # be a main title for the entire figure, indicating the dataset and organ. For the scatter plot, the color
            # of each point should be different depending on whether the point is an outlier or not.
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle("{} - {}".format(dataset, organ))
            sns.boxplot(x="area", data=info, ax=axes[0, 0])
            axes[0, 0].title.set_text("Area")
            sns.boxplot(x="height", data=info, ax=axes[0, 1])
            axes[0, 1].title.set_text("Height")
            sns.boxplot(x="width", data=info, ax=axes[1, 0])
            axes[1, 0].title.set_text("Width")
            sns.scatterplot(x="height", y="width", data=info, hue="outlier", ax=axes[1, 1])
            axes[1, 1].title.set_text("Height vs Width")
            plt.show()
