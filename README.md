# DefectVision: Intelligent Industrial Visual Anomaly Detection & Category Discovery

[![Python 3.10 - 3.14](https://img.shields.io/badge/python-3.10%20--%203.14-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Dashboard: Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Dataset: MVTec AD](https://img.shields.io/badge/dataset-MVTec%20AD-green.svg)](https://www.mvtec.com/company/research/datasets/mvtec-ad)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](license.txt)

## Project Overview
**DefectVision** is an industrial computer vision and unsupervised machine learning system engineered for automated quality inspection and visual category discovery using the **MVTec Anomaly Detection (MVTec AD)** benchmark. 

The system learns the distribution of normal product appearance strictly from pristine training images (`train/good`) without any defect supervision. Deep multi-scale representations are extracted from intermediate CNN layers, indexed into patch memory banks, scored via nearest-neighbor distance metrics, localized with 2D Gaussian-smoothed heatmaps, and explored across unsupervised manifold embeddings (**PCA**, **t-SNE**, **K-Means**, **DBSCAN**).

> **Repository Note:** To comply with GitHub's file size limits, raw MVTec datasets (`data/mvtec/`, ~5.26 GB) and large serialized model weights (`models/*.pkl`, ~400 MB) are excluded via `.gitignore`. Directory anchors (`.gitkeep`), evaluation plots (`results/plots/`), and benchmark reports (`reports/`, `results/metrics/`) are tracked in the repository.

---

## 🏆 Benchmark Performance Scoreboard

These metrics were produced by actual benchmark runs executed on Google Colab with an NVIDIA T4 GPU:

| Category | Train Normal | Test Normal | Test Defects | AUROC | AUPRC | F1-Score | Precision | Recall | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bottle** | 209 | 20 | 63 | **99.68%** | **99.90%** | **0.9756** | 100.00% | 95.24% | 35.8783 |
| **Cable** | 224 | 58 | 92 | **83.08%** | **88.28%** | **0.8377** | 80.81% | 86.96% | 39.5373 |
| **Hazelnut** | 391 | 40 | 70 | **99.36%** | **99.58%** | **0.9929** | 98.59% | 100.00% | 39.5638 |

### Evaluation Methodology & Dataset Splits
* **Dataset Splits:** Trained strictly on pristine `train/good` normal images (zero defect supervision). Evaluated on official MVTec AD test sets containing both normal samples and real industrial defect modes.
* **Patch Feature Representation:** Multi-scale patch features (1536 dimensions) extracted from ResNet-50 intermediate Layer 2 and Layer 3 feature maps at $28 \times 28$ spatial resolution.
* **Memory Bank & Subsampling:** 10% coreset/random subsampling on normal patches; nearest-neighbor distance scoring with $k=1$.
* **Heatmap Localization:** Maximum patch distance scored per spatial location and smoothed using a 2D Gaussian filter ($\sigma = 4.0$).
* **Metrics Calculation:** Image-level AUROC and AUPRC computed using scikit-learn (`metrics.roc_auc_score`, `metrics.precision_recall_curve`, `metrics.auc`). Optimal threshold selected by maximizing the harmonic F1-score across precision-recall thresholds.

---

## 3-Person Team Division & Architecture

The DefectVision framework is organized into three parallel engineering tracks:

```
DefectVision/
├── app/
│   └── streamlit_app.py                    # Production Streamlit inspection & analytics app (3-tab HUD)
├── authentication/
│   ├── config/
│   │   └── settings.py                     # Auth configuration & role definitions
│   ├── database/
│   │   └── auth_users.db                   # [Excluded in Git] SQLite user store (WAL mode)
│   ├── auth_backend.py                     # bcrypt password hashing & user validation
│   ├── login.py                            # Login portal UI
│   ├── register.py                         # New user registration UI
│   ├── forgot_password.py                  # Password reset UI
│   ├── session_manager.py                  # Session state & remember-me logic
│   ├── user_dashboard.py                   # Operator/engineer dashboard
│   ├── admin_dashboard.py                  # Admin management panel
│   ├── styles.css                          # Auth portal custom CSS
│   └── requirements-auth.txt              # Auth-specific dependencies
├── .streamlit/
│   ├── config.toml                         # Streamlit server and UI configurations
│   └── credentials.toml                   # Streamlit client configurations
├── data/
│   ├── mvtec/                              # [Excluded in Git] MVTec categories (bottle, cable, hazelnut)
│   │   ├── bottle/  (train/, test/, ground_truth/)
│   │   ├── cable/   (train/, test/, ground_truth/)
│   │   └── hazelnut/(train/, test/, ground_truth/)
│   └── embeddings/
│       └── metadata.csv                   # Master image metadata index (1,167 samples)
├── models/
│   ├── .gitkeep                            # Directory anchor (tracked in Git)
│   ├── memory_bank_bottle.pkl              # [Excluded in Git] Bottle PatchCore memory bank
│   ├── memory_bank_cable.pkl               # [Excluded in Git] Cable PatchCore memory bank
│   └── memory_bank_hazelnut.pkl            # [Excluded in Git] Hazelnut PatchCore memory bank
├── notebooks/
│   ├── 01_dataset_exploration.ipynb        # Dataset EDA & sample visualization
│   ├── 01_clustering_pipeline_walkthrough.ipynb  # Chandan's clustering walkthrough
│   ├── 02_feature_extraction_and_anomaly_detection.ipynb  # Tanya Tejaswi's training notebook
│   └── 03_pca_and_clustering.ipynb         # Chandan's PCA, t-SNE, K-Means & DBSCAN notebook
├── reports/
│   └── anomaly_detection_summary.csv       # Master multi-category performance scoreboard
├── results/
│   ├── metrics/                            # Category-level JSON evaluation metrics
│   │   ├── bottle_metrics.json
│   │   ├── cable_metrics.json
│   │   ├── hazelnut_metrics.json
│   │   └── clustering_benchmark.json       # Chandan's clustering evaluation metrics
│   └── plots/                              # Evaluation curves & defect localization samples
│       ├── bottle_roc_pr.png
│       ├── bottle_sample_0.png, 1.png, 2.png
│       ├── cable_roc_pr.png
│       ├── cable_sample_0.png, 1.png, 2.png
│       ├── hazelnut_roc_pr.png
│       ├── hazelnut_sample_0.png, 1.png, 2.png
│       ├── pca_scree.png                   # PCA explained variance scree plot
│       ├── tsne_kmeans.png                 # t-SNE coloured by K-Means cluster
│       ├── tsne_dbscan.png                 # t-SNE coloured by DBSCAN labels
│       └── kmeans_elbow.png                # K-Means elbow curve
├── scripts/
│   └── generate_sample_images.py           # Utility to generate dataset sample previews
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                    # Dataset loaders & image normalization
│   ├── feature_extraction.py               # ResNet-50 patch and global feature extractor
│   ├── anomaly_detection.py                # PatchCore memory bank & inference engine
│   ├── dimensionality_reduction.py         # PCA scree analysis & t-SNE projections
│   ├── clustering.py                       # K-Means elbow, DBSCAN & outlier detection
│   ├── visualization.py                    # Heatmap smoothing & JET overlay rendering
│   └── evaluation.py                       # Image-level AUROC, AUPRC & clustering metrics
├── tests/
│   ├── test_anomaly_detection.py
│   ├── test_clustering.py
│   ├── test_dimensionality_reduction.py
│   ├── test_evaluation.py
│   ├── test_feature_extraction.py
│   └── test_visualization.py
├── run_tanya_pipeline.py                   # Tanya Tejaswi's headless CLI batch training pipeline
├── run_pipeline.py                         # Shared pipeline runner (Gayatri U & Tanya Tejaswi)
├── pytest.ini                              # pytest configuration
├── requirements.txt
├── .gitignore
├── license.txt                             # CC BY-NC-SA 4.0 License
└── README.md
```

### 1. Member 1: Tanya Tejaswi (Computer Vision Specialist)
- **MVTec AD Ingestion**: Strict zero-defect supervision across bottle, cable, and hazelnut categories.
- **Deep Feature Representation**: Intermediate ResNet-50 Layer 2 & Layer 3 feature fusion.
- **Memory Bank & Scoring**: Nearest-neighbor patch distance scoring, coreset subsampling, Gaussian heatmap smoothing ($\sigma=4.0$), and JET overlays.
- **Evaluation & Metrics**: AUROC (99.68% bottle, 99.36% hazelnut), AUPRC (99.90%), F1 optimization, and model serialization (`models/memory_bank_{category}.pkl`).

### 2. Member 2: Chandan (Unsupervised Learning & Dimensionality Reduction)
- **Feature Scaling**: Standard normalization across 2048-dimensional feature vectors (`src/dimensionality_reduction.py`).
- **PCA Decomposition**: Scree plots, cumulative explained variance analysis (90% and 95% retention).
- **t-SNE Manifold Learning**: Non-linear 2D projections across perplexity settings (15, 30, 50).
- **Unsupervised Clustering**: K-Means elbow/silhouette optimization, DBSCAN density-based noise outlier detection (`src/clustering.py`).
- **Validation Metrics**: Evaluation via Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI).

### 3. Member 3: Gayatri U (Full-Stack MLOps & Dashboard Application)
- **Production Dashboard**: 3-Tab interactive Streamlit application (`app/streamlit_app.py`) running on port 8501:
  - **Tab 1: Live Defect Inspector & Benchmarks**: Real-time image upload or sample picker, PatchCore memory bank querying, dynamic threshold slider, verdict badge (NORMAL vs DEFECT DETECTED), 3-panel visualization (Original, Heatmap, Overlay), Tanya Tejaswi's official scoreboard, and evaluation curves (ROC/PR).
  - **Tab 2: Interactive Plotly Cluster Explorer**: 2D/3D PCA & t-SNE scatter plots with hover metadata tooltips and category filtering.
  - **Tab 3: Visual Cluster Centroid Gallery**: Automatic calculation of K-Means centroids and display of top 5 nearest visual exemplar products per cluster.
- **MLOps Architecture**: Sub-50ms query latency via `@st.cache_resource` for PyTorch models and `@st.cache_data` for metadata indices.

---

## Environment Setup & Execution

### 1. Virtual Environment Activation
* **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Production Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

### 4. Run the Test Suite
```bash
pytest -v
```

---

## License & Acknowledgements
- Distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**.
- Benchmark dataset provided by **MVTec Software GmbH** (Bergmann et al., CVPR 2019).
