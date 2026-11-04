# Unsupervised Visual Anomaly Discovery for Industrial Quality Inspection

## Project Overview
This project develops an industrial computer vision and unsupervised anomaly discovery system using the **MVTec Anomaly Detection (MVTec AD)** benchmark. The system learns normal product appearance strictly from pristine training images (`train/good`), extracts deep representations, discovers visual patterns and clusters via dimensionality reduction (**PCA**, **t-SNE**), detects anomalous regions, and generates pixel-level inspection heatmaps.

### Syllabus & Course Outcome Alignment
- **Course Topics**: Unit IV & II — Dimensionality Reduction (PCA / t-SNE) and Clustering.
- **Course Outcomes**: CO2, CO3, CO4.
- **Core Methodology**: Pretrained CNN feature extraction $\rightarrow$ PCA explained variance $\rightarrow$ t-SNE 2D manifold visualization $\rightarrow$ Unsupervised Clustering (K-Means, DBSCAN, HDBSCAN) $\rightarrow$ Memory Bank Anomaly Scoring & Heatmaps.

---

## Team Division & Deliverables

### 1. Tanya — Computer Vision & Anomaly Detection (Lead: Pipeline B)
- **Dataset Preparation & Preprocessing**: Standardized OpenCV / torchvision pipelines.
- **Deep Feature Extraction**: Pretrained CNN backbone (ResNet-50) extracting global image embeddings and multi-scale patch features.
- **Deliverable to Gayathri**: Exported image embeddings (`image_embeddings.npy`) + metadata (`metadata.csv`).
- **Memory Bank & Anomaly Scorer**: PatchCore-style memory bank fitted strictly on normal training data.
- **Inspection Heatmaps**: Anomaly localization maps and visual overlay blends.
- **Quantitative Evaluation**: Image-level AUROC & AUPRC, pixel-level localization metrics.

### 2. Gayathri — Unsupervised Learning & Dimensionality Reduction (Lead: Pipeline A)
- **Feature Scaling**: Standardization / L2 normalization of extracted embeddings.
- **PCA**: Scree plots, cumulative explained variance analysis, component selection.
- **t-SNE**: 2D visualization of object categories and defect distributions across perplexity settings.
- **Clustering**: K-Means, DBSCAN, and HDBSCAN clustering without label guidance.
- **Evaluation**: Silhouette score, Davies-Bouldin index, and post-hoc reference evaluation (ARI / NMI).
- **Cluster Interpretation**: Profile analysis, representative images, and responsible reporting of limitations.

### Shared Deliverables
- Interactive **Streamlit Dashboard** (`app/streamlit_app.py`)
- Technical report, evaluation logs, and video demonstration.

---

## Project Structure

```
industrial-anomaly-discovery/
├── data/
│   ├── mvtec/                     # Extracted categories (bottle, cable, hazelnut)
│   └── embeddings/                # Tanya's exported features for Gayathri
│       ├── image_embeddings.npy
│       └── metadata.csv
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_feature_extraction_and_anomaly_detection.ipynb
│   ├── 03_pca_analysis.ipynb
│   └── 04_clustering_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Dataset loaders & OpenCV/torchvision transforms
│   ├── feature_extraction.py      # ResNet feature extractor & embedding exporter
│   ├── anomaly_detection.py       # PatchCore memory bank & nearest-neighbor detector
│   ├── visualization.py           # Heatmap overlays & multi-panel plots
│   └── evaluation.py              # AUROC, AUPRC, pixel AUROC & curve plots
├── app/
│   └── streamlit_app.py           # Interactive inspection & clustering dashboard
├── results/
│   ├── plots/                     # Generated heatmaps, ROC curves, cluster plots
│   └── metrics/                   # Saved JSON/CSV performance benchmarks
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### Option A: Locally in VS Code
1. Open the project folder in VS Code:
   ```bash
   code .
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open any notebook in `notebooks/` directly with the VS Code Jupyter extension.
4. Or run modules via Python CLI:
   ```bash
   python src/preprocessing.py
   ```

### Option B: In Google Colab
1. Upload the project folder or clone your GitHub repository into Google Colab.
2. Ensure runtime hardware accelerator is set to **GPU** (`Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`).
3. Open `notebooks/01_dataset_exploration.ipynb` or `notebooks/02_feature_extraction_and_anomaly_detection.ipynb`.
4. Run all cells — the code automatically detects Google Colab and configures paths and GPU acceleration.
