# DefectVision 2.0: Autonomous Optical Inspection & Defect Discovery Platform

[![Live Demo: Render](https://img.shields.io/badge/Live%20Demo-Render%20Cloud-success.svg)](https://defectvision-intelligent-industrial.onrender.com)
[![Python 3.10 - 3.14](https://img.shields.io/badge/python-3.10%20--%203.14-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch%20(CPU%20%2F%20CUDA)-EE4C2C.svg)](https://pytorch.org/)
[![Platform: FastAPI & Tailwind HUD](https://img.shields.io/badge/platform-FastAPI%20%26%20Tailwind%20HUD-00B050.svg)](https://defectvision-intelligent-industrial.onrender.com)
[![Datasets: VisA, MVTec AD, NEU Steel](https://img.shields.io/badge/datasets-VisA%20%7C%20MVTec%20%7C%20NEU-green.svg)](https://github.com/TanyaTejaswi/DefectVision-Intelligent-Industrial-Anomaly-Detection)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](license.txt)

> 🌐 **Live Web Application**: [https://defectvision-intelligent-industrial.onrender.com](https://defectvision-intelligent-industrial.onrender.com)

---

## 📌 Project Overview
**DefectVision 2.0** is an enterprise-grade industrial computer vision and unsupervised anomaly detection platform engineered for automated quality assurance and sub-millimeter visual defect discovery across **8 critical manufacturing lines**.

Operating on **PatchCore memory bank architectures**, the system learns the distribution of pristine product appearance strictly from normal baseline imagery without requiring defect supervision. Intermediate ResNet-50 Layer 2 & Layer 3 multi-scale representations are indexed into memory banks, queried via fast nearest-neighbor distance metrics, localized with 2D Gaussian heatmaps, and displayed on a modern **Dual Comparative Optical Viewfinder** with automated conveyor routing directives.

---

## 🏆 8-Line Industrial Suite Benchmark Scoreboard

Official benchmarks validated against the calibrated decision thresholds:

| Manufacturing Line | Category Code | AUROC | AUPRC | F1-Score | Precision | Recall | Calibrated Threshold | Dataset Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Micro-Electronics Trace** | `pcb1` | **95.05%** | **98.89%** | **0.9700** | 97.00% | 97.00% | 46.80 | VisA Benchmark |
| **Micro-Soldering Assembly** | `pcb2` | 70.50% | **92.46%** | **0.9091** | 83.33% | 100.00% | 39.50 | VisA Benchmark |
| **Multi-Layer Logic Board** | `pcb3` | 69.25% | **91.91%** | **0.9167** | 85.34% | 99.00% | 43.50 | VisA Benchmark |
| **High-Density Connector** | `pcb4` | 76.35% | **94.43%** | **0.9174** | 84.75% | 100.00% | 41.20 | VisA Benchmark |
| **Pharmaceutical Capsules** | `capsules` | 80.25% | **93.24%** | **0.9100** | 88.00% | 94.00% | 41.00 | MVTec AD |
| **Glass Beverage Containers** | `bottle` | **99.68%** | **99.50%** | **0.9756** | 100.00% | 95.24% | 36.00 | MVTec AD |
| **Automotive Wire Harness** | `cable` | 83.08% | 84.20% | **0.8377** | 82.00% | 85.60 | 41.50 | MVTec AD |
| **Hot-Rolled Steel Strip** | `steel` | **100.00%** | **100.00%** | **1.0000** | 100.00% | 100.00% | 35.00 | NEU Surface Defect |

---

## ⚡ Key Platform Capabilities

1. **Dual Comparative Optical Viewfinder**:
   - **Viewfinder 1 (Current Raw Unit)**: Direct high-resolution sensor feed overlaid with AI bounding boxes (`PASS: SMT ARRAY`, `DEFECT: PINCH`, or `CRITICAL: FOREIGN OBJECT`).
   - **Viewfinder 2 (Defect Localization)**: Interactive thermal pseudo-coloring (**JET**, **Turbo**, **Inferno**, **Viridis**) with real-time opacity blending and toggleable Raw Sensor view.
2. **Out-of-Domain Foreign Object Rejection**:
   - Automated cross-correlation validation against line-specific Golden CAD references.
   - Prevents non-industrial objects, personal webcam pictures, or fixture mismatches from passing; flags them with high anomaly scores and conveyor hazard ejection.
3. **Curated Standalone Operation**:
   - Includes 188 curated normal/defect reference samples and 62 ground-truth defect masks (`data/curated/`) alongside 8 pre-trained float16 memory banks (`models/`). Runs out of the box with zero external dataset downloads required.
4. **4 Flexible Ingestion Channels**:
   - **Local File Upload**: Drag-and-drop any component image from disk.
   - **Line Dataset Selector**: 1-click test against verified factory samples.
   - **Live Hardware Camera**: Direct webcam capture from factory workstations.
   - **Web / Google Image URL**: Instant URL paste or curated online exemplar grid.

---

## 🗂️ Clean Production Architecture

```text
DefectVision/
├── api/                                    # Vercel serverless integration entrypoint
│   └── index.py
├── api_server.py                           # Production FastAPI AOI Server & API endpoints
├── data/
│   ├── curated/                            # Standalone 8-line sample images & defect masks
│   │   ├── bottle/                         # (good/, anomaly/, masks/)
│   │   ├── cable/                          # (good/, anomaly/, masks/)
│   │   ├── capsules/                       # (good/, anomaly/, masks/)
│   │   ├── pcb1/, pcb2/, pcb3/, pcb4/      # (good/, anomaly/, masks/)
│   │   └── steel/                          # (good/, anomaly/, masks/)
│   └── embeddings/                         # Embedding metadata index
├── frontend/                               # DefectVision 2.0 Web Platform & Instrumental HUD
│   ├── index.html                          # Single-page AOI Studio application
│   ├── app.js                              # Client SPA orchestration & viewfinder logic
│   ├── auth.js                             # Workstation operator authentication manager
│   ├── telemetry.js                        # Telemetry charts & metrics widgets
│   └── plots/                              # Pre-rendered manifold & ROC evaluation curves
├── models/                                 # 8 Pre-trained float16 PatchCore memory banks
│   ├── memory_bank_bottle.pkl
│   ├── memory_bank_cable.pkl
│   ├── memory_bank_capsules.pkl
│   ├── memory_bank_pcb1.pkl ... pcb4.pkl
│   └── memory_bank_steel.pkl
├── notebooks/                              # Research & EDA exploratory Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 01_clustering_pipeline_walkthrough.ipynb
│   ├── 02_feature_extraction_and_anomaly_detection.ipynb
│   └── 03_pca_and_clustering.ipynb
├── reports/                                # Dynamic benchmark calibration summary CSV
│   └── anomaly_detection_summary.csv
├── results/                                # Category evaluation metrics and ROC curves
│   ├── metrics/                            # Per-line JSON benchmark metrics
│   └── plots/                              # Evaluation curves & cluster projections
├── scripts/                                # Offline training pipelines and utilities
│   ├── run_pipeline.py
│   ├── run_pipeline_m1.py
│   ├── train_all_categories.py
│   ├── train_batch.py
│   └── train_custom_lines.py
├── src/                                    # Core ML & Vision algorithms
│   ├── anomaly_detection.py                # PatchCore nearest-neighbor inference engine
│   ├── feature_extraction.py               # ResNet-50 patch feature representation
│   ├── preprocessing.py                    # Domain calibration & normalization
│   ├── dimensionality_reduction.py         # PCA & t-SNE manifold projections
│   ├── clustering.py                       # K-Means & DBSCAN cluster optimization
│   ├── evaluation.py                       # AUROC, AUPRC & F1 calculations
│   └── visualization.py                    # 2D Gaussian heatmap rendering
├── tests/                                  # Comprehensive unit test suite & runner
│   ├── pytest.ini
│   ├── run_tests.py
│   └── test_*.py
├── .dockerignore                            # Docker build ignore rules
├── .gitignore                              # Git exclusion rules
├── Dockerfile                              # Production container build specification
├── requirements.txt                        # Production dependencies
├── vercel.json                             # Cloud CDN rewrite configuration
├── license.txt                             # CC BY-NC-SA 4.0 License
└── README.md
```

---

## ⚙️ Core Technical Modules & Functionality

### 1. Computer Vision & PatchCore Anomaly Localization Engine (`src/`)
- **Deep Feature Representation**: Intermediate ResNet-50 multi-layer patch feature extraction across VisA, MVTec AD, and NEU Steel benchmarks.
- **Memory Bank & Spatial Scoring**: Nearest-neighbor patch distance querying, 2D Gaussian heatmap smoothing ($\sigma = 4.0$), and dynamic thermal colormaps (JET, Turbo, Inferno, Viridis).
- **Curated Dataset & Defect Masks**: Standalone 8-line offline dataset bundling (`data/curated/`) and ground-truth pixel-level defect mask mapping.
- **Decision Threshold Calibration**: Harmonic F1-score optimization across all 8 production lines for automated PASS / DEFECT conveyor decisions.

### 2. Unsupervised Learning & Manifold Analysis (`src/dimensionality_reduction.py`, `src/clustering.py`)
- **Dimensionality Reduction**: Standardized feature normalization and PCA explained variance scree decomposition (90% & 95% retention).
- **Manifold Learning & Clustering**: t-SNE non-linear embeddings, K-Means elbow optimization, and DBSCAN density-based outlier detection.
- **Quantitative Validation**: Evaluation via Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and category manifold projections.
- **Serverless Integration**: Vercel serverless integration (`api/index.py`, `vercel.json`) with lean production requirements.

### 3. Full-Stack AOI Platform & Infrastructure (`frontend/`, `api_server.py`)
- **Instrumental HUD**: Single-Page Application (SPA) with Dual Comparative Optical Viewfinder, real-time blend controls, and telemetry dashboards.
- **API Server & Routing**: High-throughput FastAPI + Uvicorn server (`api_server.py`) with sub-50ms inference latency.
- **Workstation Security & Auth**: Operator Authentication Portal (`frontend/auth.js`) supporting role-based access, credential verification, and user registration.
- **Lean Containerization**: Production [Dockerfile](Dockerfile) and [.dockerignore](.dockerignore) optimized for memory-constrained CPU cloud environments (Render / Hugging Face Spaces).

---

## 🚀 Quickstart Guide

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/TanyaTejaswi/DefectVision-Intelligent-Industrial-Anomaly-Detection.git
cd DefectVision-Intelligent-Industrial-Anomaly-Detection

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Local Inspection Platform
```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at **`http://127.0.0.1:8000`** (or `http://localhost:8000`).

### 3. Run Test Suite
```bash
python tests/run_tests.py
```

### 4. Cloud Deployment (Live on Render)
DefectVision 2.0 is live and deployed on Render's container runtime:
- 🔗 **Live Web URL**: [https://defectvision-intelligent-industrial.onrender.com](https://defectvision-intelligent-industrial.onrender.com)
- **Deployment Type**: Docker Web Service (CPU-only, memory-optimized with single-model LRU caching < 200 MB).
- **Auto-Sync**: Automatically synchronized with the `main` branch.

---

## 📜 License & Citation
- Distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**.
- Benchmark datasets provided by **MVTec Software GmbH** (Bergmann et al., CVPR 2019), **VisA Dataset** (Zou et al., 2022), and **NEU Surface Defect Database** (Song & Yan, 2013).
