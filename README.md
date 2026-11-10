# DefectVision: Master Engineering Blueprint

**Industrial Visual Anomaly Detection & Visual Category Discovery**  
*A 3-Person Parallel Engineering Architecture: Machine Learning [ML] • Back-End Engine [BE] • Front-End UI [FE]*

---

## 1. Project Overview & Architectural Taxonomy
DefectVision is an enterprise-grade industrial inspection framework built to detect rare manufacturing anomalies and discover unsupervised visual clusters without requiring defect annotations during training.

The system is organized into three distinct tiers:
- **[ML] Machine Learning Layer**: Deep representations (ResNet-50 Layer 2+3), PatchCore patch memory bank, k-NN anomaly distance scoring, PCA eigenvalue decomposition, t-SNE manifold projections, K-Means clustering, and AUROC/AUPRC evaluation.
- **[BE] Back-End Engine Layer**: MVTec dataset ingestion, file traversal, PyTorch DataLoader pipelines, ImageNet normalization, binary feature caching, disk I/O, memory bank pickle serialization, and inference wrapper service.
- **[FE] Front-End UI Layer**: Interactive Streamlit web application running on port 8501, real-time image upload, dynamic defect threshold sliders, heatmap opacity blending, interactive Plotly scatter exploration, and visual cluster galleries.

---

## 2. 3-Person Parallel Execution Model

| Member | Engineering Role | Primary Tier | Assigned Work Units | Status | Key Deliverable Outputs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Member 1: Tanya (Lead)** | Computer Vision & Anomaly Detection | 60% [ML] • 40% [BE] | 12 Tasks (Phase 1 & 2) | **100% [DONE]** | [`src/preprocessing.py`](src/preprocessing.py), [`src/feature_extraction.py`](src/feature_extraction.py), [`src/anomaly_detection.py`](src/anomaly_detection.py), [`src/visualization.py`](src/visualization.py), [`src/evaluation.py`](src/evaluation.py), `models/*.pkl`, `reports/anomaly_detection_summary.csv` |
| **Member 2** | Unsupervised ML & Dimensionality Reduction | 85% [ML] • 15% [BE] | 6 Tasks (Phase 3) | **[TO-DO]** | [`src/dimensionality_reduction.py`](src/dimensionality_reduction.py), [`src/clustering.py`](src/clustering.py), `data/embeddings/pca_tsne_coords.csv`, `reports/clustering_metrics.csv`, `notebooks/03_pca_and_clustering.ipynb` |
| **Member 3: Gayathri (You)** | Full-Stack MLOps & Dashboard Application | 70% [FE] • 30% [BE] | 5 Tasks (Phase 4) | **100% [DONE]** | [`app/streamlit_app.py`](app/streamlit_app.py), [`.streamlit/config.toml`](.streamlit/config.toml), [`requirements.txt`](requirements.txt), [`README.md`](README.md) |

---

## 3. Official Multi-Category Anomaly Detection Benchmark Scoreboard (Tanya's Results)

Tanya evaluated PatchCore memory bank anomaly detection across all target categories on Google Colab (T4 GPU), trained strictly on normal images (`train/good`) with zero defect supervision:

| Target Category | Train Normal | Test Normal | Test Defects | AUROC | AUPRC | F1-Score | Precision | Recall | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bottle** | 209 | 20 | 63 | **99.68%** | **99.90%** | **0.9756** | 100.00% | 95.24% | 35.8783 |
| **Cable** | 224 | 58 | 92 | **83.08%** | **88.28%** | **0.8377** | 80.81% | 86.96% | 39.5373 |
| **Hazelnut** | 391 | 40 | 70 | **99.36%** | **99.58%** | **0.9929** | 98.59% | 100.00% | 39.5638 |

---

## 4. Repository Structure

```text
DefectVision/
├── app/
│   └── streamlit_app.py            # Production Streamlit inspection & analytics app (Member 3)
├── .streamlit/
│   └── config.toml                 # Streamlit server & UI theme configurations (Member 3)
├── data/
│   ├── mvtec/                      # [Excluded in Git] MVTec categories (bottle, cable, hazelnut)
│   └── embeddings/
│       ├── metadata.csv            # Master image metadata index (1,167 samples) (Member 1)
│       └── pca_tsne_coords.csv     # 2D/3D projection coordinates (Member 2)
├── models/
│   ├── .gitkeep                    # Directory anchor
│   └── memory_bank_{cat}.pkl       # [Excluded in Git] Precomputed memory banks (Member 1)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_feature_extraction_and_anomaly_detection.ipynb
│   └── 03_pca_and_clustering.ipynb
├── reports/
│   ├── anomaly_detection_summary.csv # Master multi-category performance scoreboard (Member 1)
│   └── clustering_metrics.csv        # Academic validation metrics (Member 2)
├── results/
│   ├── metrics/                    # Category-level JSON evaluation metrics
│   └── plots/                      # ROC/PR curves & defect localization plots
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Dataset loaders & image normalization (Member 1)
│   ├── feature_extraction.py       # ResNet-50 patch & global feature extractor (Member 1)
│   ├── anomaly_detection.py        # PatchCore memory bank & inference engine (Member 1)
│   ├── dimensionality_reduction.py  # PCA & feature reduction routines (Member 2)
│   ├── clustering.py               # K-Means, DBSCAN, and HDBSCAN (Member 2)
│   ├── visualization.py            # Heatmap smoothing & JET overlay rendering (Member 1)
│   └── evaluation.py               # Image-level AUROC, AUPRC & curve plotting (Member 1)
├── requirements.txt                # Unified project dependencies (Member 3)
├── .gitignore                      # Git exclusion rules
├── license.txt                     # CC BY-NC-SA 4.0 License
└── README.md                       # Master Engineering Blueprint documentation
```

---

## 5. Member 3 (Full-Stack MLOps) Implementation Details

### Tab 1: Live Defect Inspector
- **Model Loader Service**: Loads pre-trained PatchCore memory banks from `models/memory_bank_{cat}.pkl` using `@st.cache_resource` for sub-50ms inference.
- **Dual Ingestion**: Real-time custom image upload (PNG/JPG) or instant sample selection from `data/embeddings/metadata.csv`.
- **Dynamic Decision Controls**: Threshold slider (defaulting to the optimal Youden/F1 cutoff) and heatmap opacity blending ($\alpha$).
- **Visual Triple**: Displays the Original Product Photo, Spatial Anomaly Heatmap (JET colormap), and Defect Localization Overlay side-by-side with verdict badges (`NORMAL (PASS)` vs `DEFECT DETECTED`).
- **Benchmark Scorecard**: Live multi-category summary loaded from `reports/anomaly_detection_summary.csv`.

### Tab 2: Interactive Plotly Cluster Explorer
- **Manifold Ingestion**: Ingests `data/embeddings/pca_tsne_coords.csv` from Member 2 (with resilient fallback).
- **Interactive Projections**: Toggle between **t-SNE (2D)**, **PCA (2D)**, and **PCA (3D)**.
- **Dynamic Filtering**: Filter by category (`bottle`, `cable`, `hazelnut`) and color points by `cluster_kmeans`, `category`, or `defect_type` with full hover tooltips (`image_id`, `category`, `defect_type`).

### Tab 3: Visual Cluster Centroid Gallery
- **Exemplar Discovery**: For each K-Means cluster ($K=3$), calculates the Euclidean cluster centroid in PCA space and auto-displays the **5 nearest centroid images** in responsive columns.
- **Visual Mode Interpretation**: Enables inspection engineers to visually interpret what each discovered cluster represents.

---

## 6. How to Run the Production Dashboard

```powershell
# 1. Activate Virtual Environment
.venv\Scripts\activate

# 2. Run the DefectVision Streamlit Dashboard
streamlit run app/streamlit_app.py
```
The application will open live on `http://localhost:8501`.
