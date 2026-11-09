# DefectVision: Intelligent Industrial Visual Anomaly Detection & Category Discovery

[![Python 3.10 - 3.12](https://img.shields.io/badge/python-3.10%20--%203.12-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Dashboard: Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Dataset: MVTec AD](https://img.shields.io/badge/dataset-MVTec%20AD-green.svg)](https://www.mvtec.com/company/research/datasets/mvtec-ad)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](license.txt)

## Project Overview
**DefectVision** is an industrial computer vision and unsupervised machine learning system engineered for automated quality inspection and visual category discovery using the **MVTec Anomaly Detection (MVTec AD)** benchmark. 

The system learns the distribution of normal product appearance strictly from pristine training images (`train/good`) without any defect supervision. Deep multi-scale representations are extracted from intermediate CNN layers, indexed into patch memory banks, scored via nearest-neighbor distance metrics, localized with 2D Gaussian-smoothed heatmaps, and explored across unsupervised manifold embeddings (**PCA**, **t-SNE**, **K-Means**, **DBSCAN**).

> **Important Repository Note:** To comply with GitHub's file size limits, the raw MVTec dataset (`data/mvtec/`, ~5.26 GB) and large serialized model weights (`models/*.pkl`, ~400 MB) are excluded via `.gitignore`. The directory anchors (`.gitkeep`), evaluation plots (`results/plots/`), and benchmark reports (`reports/`, `results/metrics/`) are tracked in the repository. Instructions for obtaining the dataset and generating or downloading the model weights are provided below.

---

## 🏆 Benchmark Performance Scoreboard

These metrics were produced by actual benchmark runs executed on Google Colab with an NVIDIA T4 GPU:

| Category | Train Normal | Test Normal | Test Defects | AUROC | AUPRC | F1-Score | Precision | Recall | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bottle** | 209 | 20 | 63 | **99.68%** | **99.90%** | **0.9756** | 100.00% | 95.24% | 35.8783 |
| **Cable** | 224 | 58 | 92 | **83.08%** | **88.28%** | **0.8377** | 80.81% | 86.96% | 39.5373 |
| **Hazelnut** | 391 | 40 | 70 | **99.36%** | **99.58%** | **0.9929** | 98.59% | 100.00% | 39.5638 |

### Evaluation Methodology & Dataset Splits
* **Dataset Splits:** Trained strictly on pristine `train/good` normal images (zero defect supervision). Evaluated on official MVTec AD test sets containing both normal samples and real industrial defect modes (broken large, contamination, crack, cut, hole, poke insulation, bent wire).
* **Patch Feature Representation:** Multi-scale patch features (1536 dimensions) extracted from ResNet-50 intermediate Layer 2 and Layer 3 feature maps at $28 \times 28$ spatial resolution.
* **Memory Bank & Subsampling:** 10% coreset/random subsampling on normal patches; nearest-neighbor distance scoring with $k=1$.
* **Heatmap Localization:** Maximum patch distance scored per spatial location and smoothed using a 2D Gaussian filter ($\sigma = 4.0$).
* **Metrics Calculation:** Image-level AUROC and AUPRC computed using scikit-learn (`metrics.roc_auc_score` and `metrics.precision_recall_curve`). The optimal decision threshold was determined by maximizing the F1-score on the precision-recall curve.
* **Preserved Reports:** Full reports and visual curves are tracked in [`reports/anomaly_detection_summary.csv`](reports/anomaly_detection_summary.csv), [`results/metrics/`](results/metrics/), and [`results/plots/`](results/plots/).

---

## Architecture & Team Division

```
                      ┌────────────────────────────────────────┐
                      │    MVTec AD Industrial Images (1,167)  │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │ Pipeline A: Anomaly Det.  │                   │ Pipeline B: Unsupervised  │
   │ [Tanya - Computer Vision] │                   │ [Member 2 - Clustering]   │
   ├───────────────────────────┤                   ├───────────────────────────┤
   │ • ResNet-50 Layer 2+3     │                   │ • Global 2048-dim vectors │
   │ • PatchCore Memory Bank   │                   │ • PCA Scree & Var (95%)   │
   │ • k-NN Anomaly Distance   │                   │ • 2D t-SNE Projections    │
   │ • 2D Gaussian Heatmaps    │                   │ • K-Means & DBSCAN        │
   │ • AUROC / AUPRC Curves    │                   │ • Silhouette / ARI / NMI  │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │   Interactive Streamlit Application    │
                      │       [Member 3 - Full-Stack]          │
                      │  Live Inspection • PCA • Scoreboard    │
                      └────────────────────────────────────────┘
```

### 1. Tanya — Computer Vision & Anomaly Detection Lead (Completed)
- **Data Ingestion & Preprocessing**: Automated MVTec dataset structure ingestion and PyTorch transforms (`src/preprocessing.py`).
- **Deep Feature Representation**: Pretrained ResNet-50 feature extractor with hooks for multi-scale spatial patches (`src/feature_extraction.py`).
- **PatchCore Anomaly Engine**: Memory bank fitted on normal features with k-NN nearest-neighbor distance scoring (`src/anomaly_detection.py`).
- **Defect Localization**: 2D Gaussian spatial smoothing ($\sigma=4.0$) and JET colormap alpha-blend defect overlays (`src/visualization.py`).
- **Model Serialization**: Exported fitted memory banks (`models/memory_bank_{category}.pkl`) for instant sub-second inference.
- **Evaluation**: Generated ROC and Precision-Recall curves across all target categories (`results/plots/`).

### 2. Member 2 — Unsupervised Learning & Dimensionality Reduction Lead
- **Feature Scaling**: Standardization and L2 normalization of 2048-dimensional feature vectors (`src/dimensionality_reduction.py`).
- **PCA Decomposition**: Scree plots, cumulative explained variance analysis, and dimensionality reduction for 90% and 95% information retention.
- **t-SNE Manifold Learning**: Non-linear 2D projection across perplexity settings (15, 30, 50) and export of `pca_tsne_coords.csv`.
- **Unsupervised Clustering**: K-Means elbow/silhouette optimization, DBSCAN density-based noise outlier detection (`src/clustering.py`).
- **Validation Metrics**: Evaluation via Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI).

### 3. Member 3 — Full-Stack MLOps & Application Lead
- **Dashboard Interface**: Multi-view Streamlit application (`app/streamlit_app.py`).
- **Tab 1: Live Defect Inspector**: Image upload, category selection, anomaly score display, threshold verdict, and heatmap overlay.
- **Tab 2: Cluster & PCA Explorer**: Interactive 2D/3D scatter exploration of high-dimensional feature embeddings.
- **Tab 3: Benchmarks & Scoreboard**: Dynamic benchmark metrics table, ROC/PR curve viewer, and CSV report export.

---

## Project Structure

```
DefectVision/
├── app/
│   └── streamlit_app.py            # Production Streamlit inspection & analytics app
├── .streamlit/
│   └── config.toml                 # Streamlit server and UI configurations
├── data/
│   ├── mvtec/                      # [Excluded in Git] MVTec categories (bottle, cable, hazelnut)
│   └── embeddings/
│       └── metadata.csv            # Master image metadata index (1,167 samples)
├── models/
│   ├── .gitkeep                    # Directory anchor (tracked in Git)
│   ├── memory_bank_bottle.pkl      # [Excluded in Git] Generated Bottle memory bank
│   ├── memory_bank_cable.pkl       # [Excluded in Git] Generated Cable memory bank
│   └── memory_bank_hazelnut.pkl    # [Excluded in Git] Generated Hazelnut memory bank
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_feature_extraction_and_anomaly_detection.ipynb  # End-to-end executed training notebook
│   └── 03_pca_and_clustering.ipynb # Unsupervised learning & clustering notebook
├── reports/
│   └── anomaly_detection_summary.csv # Master multi-category performance scoreboard
├── results/
│   ├── metrics/                    # Category-level JSON evaluation metrics
│   │   ├── bottle_metrics.json
│   │   ├── cable_metrics.json
│   │   └── hazelnut_metrics.json
│   └── plots/                      # Evaluation curves & defect localization samples
│       ├── bottle_roc_pr.png
│       ├── bottle_sample_0.png, 1.png, 2.png
│       ├── cable_roc_pr.png
│       ├── cable_sample_0.png, 1.png, 2.png
│       ├── hazelnut_roc_pr.png
│       └── hazelnut_sample_0.png, 1.png, 2.png
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Dataset loaders & image normalization
│   ├── feature_extraction.py       # ResNet-50 patch and global feature extractor
│   ├── anomaly_detection.py        # PatchCore memory bank & inference engine
│   ├── visualization.py            # Heatmap smoothing & JET overlay rendering
│   └── evaluation.py               # Image-level AUROC, AUPRC & curve plotting
├── run_tanya_pipeline.py           # Standalone headless CLI batch training pipeline
├── requirements.txt
├── .gitignore
├── license.txt                     # CC BY-NC-SA 4.0 License
└── README.md
```

---

## Environment Setup & Prerequisites

### Tested Python Versions
* **Python 3.10 – 3.12:** Recommended for full PyTorch GPU training and patch extraction.
* **Python 3.10+ (tested on Python 3.14 on Windows):** Supported for the Streamlit dashboard, metric evaluation, and CPU-based memory bank inference.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/DefectVision.git
cd DefectVision
```

### 2. Create and Activate a Virtual Environment
* **On Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```
* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Acquisition & Model Generation

### Obtaining the MVTec AD Dataset
The raw dataset is excluded from version control due to its size (5.26 GB archive). You can acquire it via either of the following methods:

* **Official Web Download:**
  Download `mvtec_anomaly_detection.tar.xz` directly from the [MVTec Research Dataset Portal](https://www.mvtec.com/company/research/datasets/mvtec-ad).
* **Direct Script / Colab Download:**
  ```bash
  wget https://www.mydrive.ch/shares/38536/38307f4039534f61bb81267d3b73d637/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
  mkdir -p data/mvtec
  tar -xf mvtec_anomaly_detection.tar.xz -C data/mvtec/
  ```

### Generating the Model Weights (`models/*.pkl`)
The serialized memory banks (`models/memory_bank_{category}.pkl`) are generated from normal training images via the ResNet-50 patch extractor.

* **Option A: Headless CLI Script (Google Colab / GPU Recommended):**
  ```bash
  python run_tanya_pipeline.py
  ```
* **Option B: Interactive Jupyter Notebook:**
  Open [`notebooks/02_feature_extraction_and_anomaly_detection.ipynb`](notebooks/02_feature_extraction_and_anomaly_detection.ipynb) in Google Colab (with a GPU runtime), run all cells, and download the resulting `models/` directory.

---

## Quickstart & Execution

### 1. Launch the Live Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open your browser at **`http://localhost:8501`**.  
*Note: View 1 (Live Image Inspection) will load the pre-trained memory banks if they are present in `models/`.*

### 2. Verify Model Loading in CLI
*(Requires `models/memory_bank_bottle.pkl` to have been generated or placed in `models/`)*:
```bash
python -c "from src.anomaly_detection import MemoryBankAnomalyDetector; d = MemoryBankAnomalyDetector.load('models/memory_bank_bottle.pkl'); print('Loaded vectors:', d.memory_bank.shape, '| Threshold:', d.optimal_threshold)"
```

### 3. Interactive Notebooks
Open [`notebooks/02_feature_extraction_and_anomaly_detection.ipynb`](notebooks/02_feature_extraction_and_anomaly_detection.ipynb) in Jupyter or Google Colab (with a T4 GPU runtime enabled) to view step-by-step feature extraction, memory bank building, and visual localization.

---

## License
This project is distributed under the terms of the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**. See [`license.txt`](license.txt) for full license details.

---

## Acknowledgements & Dataset Citation
The benchmark dataset used in this project is provided by **MVTec Software GmbH** (Copyright © 2019 MVTec Software GmbH).

If you use this dataset in your research or project, please cite:
```bibtex
@inproceedings{bergmann2019mvtec,
  author    = {Paul Bergmann and Michael Fauser and David Sattlegger and Carsten Steger},
  title     = {A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {9592--9600},
  year      = {2019}
}
```
For inquiries regarding commercial usage of the MVTec AD dataset, please contact MVTec Software GmbH.
