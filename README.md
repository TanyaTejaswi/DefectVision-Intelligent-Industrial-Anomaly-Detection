# Unsupervised Visual Anomaly Discovery for Industrial Quality Inspection

An unsupervised machine learning framework designed to inspect industrial products, discover structural or surface anomalies, and group feature representations without requiring manual defect labels during training.

---

## 1. Project Overview
Industrial manufacturing lines require high-precision visual inspection to detect defective components. Traditional supervised systems require thousands of labeled defect examples, which are costly and rare in practice. This project addresses that challenge by:
1. Extracting high-level visual features using a pre-trained deep convolutional neural network (ResNet50).
2. Reducing embedding dimensionality to mitigate the curse of dimensionality and isolate salient feature variations (PCA).
3. Clustering representations and identifying anomalous patterns via density-based and centroid-based algorithms (K-Means, DBSCAN, HDBSCAN).
4. Providing 2D/3D visual projections (t-SNE) and evaluating cluster cohesion, separation, and anomaly detection quality.

---

## 2. Team Roles & Responsibilities

| Team Member | Responsibilities |
| :--- | :--- |
| **Gayathri** | - Dimensionality Reduction (PCA, explained variance analysis)<br>- Unsupervised Clustering (K-Means, DBSCAN, HDBSCAN)<br>- Low-dimensional Embeddings & Projections (t-SNE visualization)<br>- Clustering Evaluation Metrics |
| **Tanya** | - Industrial Dataset Sourcing & Preparation<br>- Image Preprocessing & Augmentation Pipelines<br>- Deep Feature Extraction (ResNet50 backbone)<br>- Anomaly Detection Modeling & Anomaly Scoring |

---

## 3. Repository Structure

```text
industrial-anomaly-discovery/
├── .gitignore
├── README.md
├── requirements.txt
├── datasets/                 # Raw and preprocessed industrial inspection datasets
├── notebooks/                # Exploratory notebooks for data exploration and tuning
├── src/                      # Modular Python package for the project
│   ├── __init__.py           # Package initializer
│   ├── dimensionality_reduction.py  # PCA & feature reduction routines
│   ├── clustering.py         # K-Means, DBSCAN, and HDBSCAN implementations
│   ├── visualization.py      # t-SNE projections, scree plots, cluster visualizations
│   └── evaluation.py         # Metrics for clustering and anomaly detection
├── results/
│   ├── plots/                # Exported figures, t-SNE scatter plots, and graphs
│   └── metrics/              # CSV/JSON evaluation summaries and benchmark logs
├── app/                      # Interactive user interface / dashboard (Streamlit)
└── tests/                    # Unit and integration test suites (pytest)
```

---

## 4. Evaluation Framework

To maintain scientific rigor and clear boundaries, evaluation is separated into two independent categories:

### A. Clustering Evaluation Metrics (Unsupervised / Intrinsic)
Evaluates the geometric structure, cohesion, and separation of clusters in the feature space without requiring ground truth defect labels:
- **Silhouette Coefficient**: Measures how similar an object is to its own cluster compared to neighboring clusters (range: $[-1, 1]$).
- **Davies-Bouldin Index**: Evaluates the similarity between each cluster and its most similar one, where lower values signify tighter, well-separated clusters.
- **Calinski-Harabasz Index (Variance Ratio Criterion)**: Computes the ratio of between-cluster dispersion to within-cluster dispersion, where higher values indicate better-defined clusters.

### B. Anomaly Detection Metrics (Detection & Separation Performance)
Evaluates how effectively the discovered clusters, outlier scores, or distance metrics distinguish normal products from defective components against ground-truth inspection labels (when available for validation):
- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**: Measures global ranking capability across all classification thresholds.
- **PR-AUC (Precision-Recall Area Under Curve)**: Assesses performance under high class imbalance (defects are typically rare in industrial manufacturing).
- **Precision, Recall, and F1-Score**: Evaluated at chosen decision boundaries / anomaly score thresholds.

---

## 5. Getting Started

### Prerequisites
- Python 3.10+
- Recommended: Dedicated virtual environment (`venv` or `conda`)

### Installation
1. Clone the repository and navigate into the project directory:
   ```bash
   cd industrial-anomaly-discovery
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Execution Workflow Roadmap
1. **Dataset Ingestion & Preparation** (`Tanya`): Place image datasets into `datasets/` and run preprocessing.
2. **Deep Feature Extraction** (`Tanya`): Extract deep feature embeddings using pre-trained ResNet50.
3. **Dimensionality Reduction** (`Gayathri`): Project embeddings using PCA in `src/dimensionality_reduction.py`.
4. **Clustering & Outlier Identification** (`Gayathri`): Execute K-Means, DBSCAN, and HDBSCAN in `src/clustering.py`.
5. **Visualization** (`Gayathri`): Generate t-SNE projections and cluster diagrams in `src/visualization.py`.
6. **Evaluation** (`Gayathri` & `Tanya`): Calculate intrinsic clustering metrics and anomaly detection performance in `src/evaluation.py`.

