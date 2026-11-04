# Unsupervised Visual Anomaly Discovery for Industrial Quality Inspection

An end-to-end unsupervised machine learning and computer vision framework for detecting industrial defects and discovering visual cluster representations without requiring defect annotations during training.

---

## 1. Project Overview
In industrial manufacturing, defective items are rare, unpredictable, and costly to label manually. This project addresses that challenge by combining deep representation learning with unsupervised discovery:
1. **Feature Extraction (`ResNet50`)**: Transforms raw inspection images into 2,048-dimensional semantic embeddings.
2. **Dimensionality Reduction (`PCA`)**: Projects high-dimensional feature spaces down while retaining 95% of total variance, mitigating the curse of dimensionality.
3. **Unsupervised Clustering (`K-Means`, `DBSCAN`, `HDBSCAN`)**: Partitions normal variations and isolates defect outliers as noise (`label = -1`).
4. **Memory-Bank Anomaly Detection**: Scores test samples by Euclidean distance to normal reference features and generates localized spatial heatmaps.
5. **2D Visual Manifold Projections (`t-SNE`)**: Visualizes normal clusters and highlights defect points.
6. **Interactive Dashboard (`Streamlit`)**: Provides a live web application for parameter tuning, image inspection, and defect heatmap display.

---

## 2. Team Roles & Responsibilities

| Team Member | Core Focus | Implemented Modules & Deliverables |
| :--- | :--- | :--- |
| **Gayathri** | **Unsupervised Learning & Clustering** | - [`src/dimensionality_reduction.py`](src/dimensionality_reduction.py): Scaling, PCA, explained variance, optimal component analysis.<br>- [`src/clustering.py`](src/clustering.py): K-Means with Elbow method, DBSCAN, HDBSCAN, anomaly extraction.<br>- [`src/visualization.py`](src/visualization.py): 2D t-SNE projections with red anomaly markers, dual-axis scree plots, elbow curves.<br>- [`src/evaluation.py`](src/evaluation.py): Intrinsic clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz). |
| **Tanya** | **Computer Vision & Anomaly Detection** | - [`src/feature_extraction.py`](src/feature_extraction.py): Pre-trained ResNet50 backbone, image transforms, batch feature extractor.<br>- [`src/anomaly_detection.py`](src/anomaly_detection.py): Nearest-neighbor Memory Bank detector, continuous anomaly scoring, Gaussian defect heatmaps.<br>- Dataset preparation & MVTec AD ingestion. |
| **Shared** | **System Integration** | - [`run_pipeline.py`](run_pipeline.py): Orchestrates end-to-end execution and exports benchmark metrics.<br>- [`app/streamlit_app.py`](app/streamlit_app.py): Interactive Streamlit web application.<br>- [`notebooks/`](notebooks/): Exploratory Jupyter walkthrough notebooks. |

---

## 3. Repository Structure

```text
industrial-anomaly-discovery/
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
├── run_pipeline.py           # End-to-end pipeline runner
├── datasets/                 # MVTec AD images, features.npy, labels.npy
├── notebooks/                # Exploratory Jupyter notebooks
│   └── 01_clustering_pipeline_walkthrough.ipynb
├── src/                      # Modular Python package
│   ├── __init__.py
│   ├── dimensionality_reduction.py  # PCA & feature reduction routines
│   ├── clustering.py         # K-Means, DBSCAN, and HDBSCAN
│   ├── visualization.py      # t-SNE projections, scree plots, cluster figures
│   ├── evaluation.py         # Clustering and anomaly metrics
│   ├── feature_extraction.py # ResNet50 deep feature extractor
│   └── anomaly_detection.py  # Memory bank detector and defect heatmaps
├── results/
│   ├── plots/                # Exported figures (t-SNE, scree plots, elbow curves)
│   └── metrics/              # clustering_benchmark.json
├── app/
│   └── streamlit_app.py      # Interactive Streamlit dashboard
├── scripts/
│   └── generate_sample_images.py # Industrial sample image generator
└── tests/                    # Comprehensive unit test suite (27 tests)
    ├── test_dimensionality_reduction.py
    ├── test_clustering.py
    ├── test_visualization.py
    ├── test_evaluation.py
    ├── test_feature_extraction.py
    └── test_anomaly_detection.py
```

---

## 4. Evaluation Metrics Framework

Evaluation is strictly organized into two independent categories:

### A. Clustering Evaluation Metrics (Unsupervised / Intrinsic)
Evaluates cluster cohesion, compactness, and separation directly in feature space without labels:
- **Silhouette Coefficient**: $[-1.0 \text{ to } +1.0]$, higher indicates dense, well-separated clusters.
- **Davies-Bouldin Index**: Lower is better; ratio of within-cluster scatter to inter-cluster separation.
- **Calinski-Harabasz Index**: Higher is better; variance ratio criterion.
- *Noise Exclusion*: DBSCAN/HDBSCAN noise points (`-1`) are filtered out during intrinsic evaluation to avoid distorting cluster compactness scores.

### B. Anomaly Detection Metrics (Supervised Validation / Extrinsic)
Evaluates defect separation accuracy against held-out ground truth ($0$: Normal, $1$: Defective):
- **ROC-AUC**: Global ranking accuracy across all classification thresholds.
- **PR-AUC (Average Precision)**: Optimal for industrial datasets with high defect imbalance.
- **Precision, Recall, and F1-Score**: Evaluated at the decision boundary.

---

## 5. Getting Started

### 1. Environment Activation
```powershell
# Windows PowerShell
.venv\Scripts\activate
```

### 2. Run the Full Unit Test Suite (All 27 Tests)
```powershell
pytest -v
```

### 3. Generate Sample Dataset & Extract ResNet50 Features
```powershell
# 1. Create sample industrial bottle images
python scripts/generate_sample_images.py

# 2. Extract 2048-dimensional ResNet50 features
python -c "from src.feature_extraction import extract_and_save_dataset_features; extract_and_save_dataset_features('datasets/mvtec_ad/bottle')"
```

### 4. Execute the End-to-End Pipeline
```powershell
python run_pipeline.py
```
Outputs are automatically saved to `results/plots/` and `results/metrics/clustering_benchmark.json`.

### 5. Launch the Interactive Streamlit Web App
```powershell
streamlit run app/streamlit_app.py
```

---

## 6. Real Industrial Data (MVTec AD)
To test with official industrial datasets:
1. Download categories (e.g., `bottle`, `cable`, `hazelnut`) from [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
2. Extract to `datasets/mvtec_ad/<category>/`.
3. Run `python -c "from src.feature_extraction import extract_and_save_dataset_features; extract_and_save_dataset_features('datasets/mvtec_ad/<category>')"`.
