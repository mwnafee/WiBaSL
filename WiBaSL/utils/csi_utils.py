import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tslearn import metrics
from tslearn.clustering import TimeSeriesKMeans
import math
import random

from CSIKit.filters.passband import lowpass, bandpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel
from CSIKit.reader import get_reader
from CSIKit.util import csitools

# Data Cleaning & Reshaping
def interpolate_nan_inf(matrix):
    x = np.arange(0, matrix.shape[1])
    for i in range(matrix.shape[0]):
        y = matrix[i, :]
        valid = np.isfinite(y)
        if np.sum(valid) > 1:
            interpolator = interpolate.interp1d(x[valid], y[valid], kind='linear', bounds_error=False, fill_value="extrapolate")
            matrix[i, :] = interpolator(x)
    return matrix


def interpolate_nan_inf_full(matrix):
    x = np.arange(0, matrix.shape[1])
    for i in range(matrix.shape[0]):
        y = matrix[i, :]
        valid = np.isfinite(y)
        if np.sum(valid) > 1:
            interpolator = interpolate.interp1d(x[valid], y[valid], kind='linear', bounds_error=False, fill_value="extrapolate")
            matrix[i, :] = interpolator(x)
    return matrix


def reshape_csi_data(file):
    reshaped_data = []
    csi_data = file
    if len(csi_data.shape) == 4:
        time_steps, subcarriers, antenna_x, antenna_y = csi_data.shape
        num_antenna_pairings = antenna_x * antenna_y
        reshaped_csi = csi_data.reshape(time_steps, subcarriers * num_antenna_pairings)
    return reshaped_csi


def reshape_csi_data_full(file):
    reshaped_data = []
    csi_data = file
    if len(csi_data.shape) == 4:
        time_steps, subcarriers, antenna_x, antenna_y = csi_data.shape
        num_antenna_pairings = antenna_x * antenna_y
        reshaped_csi = csi_data.reshape(time_steps, subcarriers * num_antenna_pairings)
    return reshaped_csi

# CSI Loading Functions
def split_csi_to_2d_matrices(csi_data):
    if len(csi_data.shape) != 4 or csi_data.shape[2:] != (2, 2):
        raise ValueError(f"Expected input shape (time_steps, subcarriers, 2, 2), but got {csi_data.shape}")
    
    matrices = [
        csi_data[:, :, i, j]
        for i in range(csi_data.shape[2])
        for j in range(csi_data.shape[3])
    ]
    return matrices


def load_csi(file_path):
    my_reader = get_reader(file_path)
    csi_data = my_reader.read_file(file_path, scaled=True)
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
    csi_matrices = split_csi_to_2d_matrices(csi_matrix)
    csi_matrices = [interpolate_nan_inf(matrix) for matrix in csi_matrices]
    return csi_matrices


def load_csi_full(file_path):
    my_reader = get_reader(file_path)
    csi_data = my_reader.read_file(file_path, scaled=True)
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
    csi_matrix = reshape_csi_data_full(csi_matrix)
    csi_matrix = interpolate_nan_inf_full(csi_matrix)

    csi_matrix_hampel = csi_matrix.copy()
    csi_matrix_savgol = csi_matrix.copy()

    for x in range(csi_matrix.shape[1]):
        csi_matrix_hampel[:, x] = hampel(csi_matrix_hampel[:, x], 15, 1)
        csi_matrix_savgol[:, x] = savgol_filter(csi_matrix_hampel[:, x], 11, 5)

    return csi_matrix_savgol

# Visualization & Filtering
def process_and_plot_csi(file_path, end=7):
    try:
        X = load_csi(file_path)

        if not X:
            print(f"Error: Loaded empty data from {file_path}. Skipping...")
            return

        csi_matrices_hampel = []
        csi_matrices_savgol = []

        for X_1 in X:
            csi_matrix_hampel = X_1.copy()
            csi_matrix_savgol = X_1.copy()

            for x in range(X_1.shape[1]):
                csi_matrix_hampel[:, x] = hampel(csi_matrix_hampel[:, x], 15, 1)
                csi_matrix_savgol[:, x] = savgol_filter(csi_matrix_hampel[:, x], 11, 5)

            csi_matrices_hampel.append(csi_matrix_hampel)
            csi_matrices_savgol.append(csi_matrix_savgol)

        titles = ["Savitzky-Golay Filtered CSI Data"]
        data_matrices = [csi_matrices_savgol]

        for title, matrices in zip(titles, data_matrices):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for idx, ax in enumerate(axes):
                if idx < len(matrices):
                    for x in range(matrices[idx][:-end, :].shape[1]):
                        ax.plot(matrices[idx][:-end, :][:, x])
                    ax.set_title(f"{title}: Antenna pairing {idx + 1}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Amplitude")
                else:
                    ax.axis("off")

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.show()

    except ValueError as e:
        print(f"ValueError while processing file {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error while processing file {file_path}: {e}")

# DTW Visualization
def visualize_dtw(array1, array2):
    min_len = min(len(array1), len(array2))
    array1 = np.array(array1[:min_len]).reshape((-1, 1))
    array2 = np.array(array2[:min_len]).reshape((-1, 1))

    path, _ = metrics.dtw_path(array1, array2)

    plt.figure(1, figsize=(8, 8))

    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02

    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    mat = cdist(array1, array2)

    ax_gram.imshow(mat, origin='lower')
    ax_gram.axis("off")
    ax_gram.autoscale(False)
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-", linewidth=3.)

    ax_s_x.plot(np.arange(array2.shape[0]), array2, "b-", linewidth=3.)
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, array2.shape[0] - 1))

    ax_s_y.plot(-array1, np.arange(array1.shape[0]), "b-", linewidth=3.)
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, array1.shape[0] - 1))

    plt.tight_layout()
    plt.show()
  
# Clustering
def plot_clustered_series_short(mySeries):
    for i in range(len(mySeries)):
        scaler = MinMaxScaler()
        mySeries[i] = scaler.fit_transform(mySeries[i].reshape(-1, 1)).reshape(-1)

    cluster_count = 2  # Or use: math.ceil(math.sqrt(len(mySeries)))
    km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw", n_jobs=18)
    labels = km.fit_predict(mySeries)

    fig, axs = plt.subplots(cluster_count, 1, figsize=(25, cluster_count * 4))

    for row_i, label in enumerate(set(labels)):
        cluster = []
        for i in range(len(labels)):
            if labels[i] == label:
                axs[row_i].plot(mySeries[i], c="gray", alpha=0.4)
                cluster.append(mySeries[i])

        if len(cluster) > 0:
            axs[row_i].plot(np.average(np.vstack(cluster), axis=0), c="red")

        axs[row_i].set_title(f"Cluster {label}")

    plt.tight_layout()
    plt.show()
    return labels



