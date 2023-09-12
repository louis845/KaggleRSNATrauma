import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    folder = "data_hdf5_cropped"
    shape_info = pd.read_csv("data_hdf5_cropped/shape_info.csv", index_col=1)

    num_slices = []
    max_depths = []
    max_depths_per_series = {}
    for patient_id in os.listdir(folder):
        patient_folder = os.path.join(folder, patient_id)
        if not os.path.isdir(patient_folder):
            continue
        for series_id in os.listdir(patient_folder):
            series_folder = os.path.join(patient_folder, series_id)
            if not os.path.isdir(series_folder):
                continue

            z_positions = np.load(os.path.join(series_folder, "z_positions.npy"))

            assert len(z_positions.shape) == 1
            assert abs(shape_info.loc[int(series_id), "mean_slope"] - np.mean(np.diff(z_positions))) < 1e-5
            assert np.all(np.diff(z_positions) > 0) or np.all(np.diff(z_positions) < 0)

            num_slices.append(len(z_positions))
            max_depths.append(np.max(z_positions) - np.min(z_positions))
            max_depths_per_series[int(series_id)] = np.max(z_positions) - np.min(z_positions)


    # plot the number of slices per series as a box plot, along with lines indicating the max and min number of slices
    plt.figure()
    plt.boxplot(num_slices)
    plt.plot([0, 2], [np.min(num_slices), np.min(num_slices)], color="red")
    plt.plot([0, 2], [np.max(num_slices), np.max(num_slices)], color="red")
    plt.xticks([1], ["Number of slices"])
    plt.show()

    # plot the max depth per series as a box plot, along with lines indicating the max and min max depth
    plt.figure()
    plt.boxplot(max_depths)
    plt.plot([0, 2], [np.min(max_depths), np.min(max_depths)], color="red")
    plt.plot([0, 2], [np.max(max_depths), np.max(max_depths)], color="red")
    plt.xticks([1], ["Max depth"])
    plt.show()

    # find all series such that the max depth is less than 250
    max_depths = np.array(max_depths)
    print("Num series with max depth less than 250: " + str(np.sum(max_depths < 250)))
    series_less_than_250 = [series_id for series_id in max_depths_per_series if max_depths_per_series[series_id] < 250]
    print("Series less than 250: " + str(series_less_than_250))

    # find all series such that the max depth is between 250 and 300
    print("Num series with max depth between 250 and 300: " + str(np.sum((max_depths >= 250) & (max_depths < 300))))
    series_between_250_and_300 = [series_id for series_id in max_depths_per_series if
                                    max_depths_per_series[series_id] >= 250 and max_depths_per_series[series_id] < 300]
    print("Series between 250 and 300: " + str(series_between_250_and_300))

    # find all series such that the max depth is above 800
    print("Num series with max depth above 800: " + str(np.sum(max_depths > 800)))
    series_above_800 = [series_id for series_id in max_depths_per_series if max_depths_per_series[series_id] > 800]
    print("Series above 800: " + str(series_above_800))