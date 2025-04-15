# 📡 CSI Utils

**CSI Utils** is a Python toolkit designed to simplify the preprocessing, filtering, visualization, and analysis of **Channel State Information (CSI)** data. It is especially helpful for research in wireless sensing, activity recognition, and other CSI-based applications.

---

## 🚀 Key Features

- **CSI Data Loading:** Load CSI data easily in both full and antenna-pair-wise matrix formats.
- **Data Filtering:** Clean CSI signals using Hampel and Savitzky-Golay filters.
- **Visualization Tools:** 
  - Interactive line plots for visualizing CSI signals.
  - Dynamic Time Warping (DTW) visualizations for comparing CSI time series.
- **Time-Series Clustering:** Perform clustering on CSI data using DTW-based KMeans.
- **Interpolation & Cleaning:** Automatically handle NaN and infinite values through linear interpolation.

---

## 📂 Repository Structure

CSI-Utils/
│
├── csi_utils.py        # Core CSI processing functions
├── README.md           # Documentation (you are here!)
├── example_notebooks/  # (Optional) Example Jupyter notebooks
└── data/               # (Optional) Sample CSI data files

---

## 📦 Installation & Requirements

Install the required libraries easily with `pip`:

```bash
pip install numpy matplotlib scipy scikit-learn tslearn
pip install CSIKit
