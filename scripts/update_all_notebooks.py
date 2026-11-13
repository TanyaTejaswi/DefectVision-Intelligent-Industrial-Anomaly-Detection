"""
Script to update all 4 Jupyter Notebooks in notebooks/ to reflect the 8-Line Industrial Suite:
1. 01_dataset_exploration.ipynb
2. 02_feature_extraction_and_anomaly_detection.ipynb
3. 03_pca_and_clustering.ipynb
4. 01_clustering_pipeline_walkthrough.ipynb
"""

import os
import sys
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_03_pca_and_clustering():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell("""# Unsupervised Visual Anomaly Discovery for Industrial Quality Inspection
## Comprehensive Manifold & Clustering Walkthrough: 8-Line Industrial Suite
**Author:** Gayatri U & Tanya Tejaswi (Unsupervised Learning & Dimensionality Reduction)

This notebook demonstrates the end-to-end unsupervised pipeline across all **8 Industrial Manufacturing Lines**:
1. **pcb1** (Mainboard Traces - Amazon VisA)
2. **pcb2** (Micro-Soldering - Amazon VisA)
3. **pcb3** (IC Component Array - Amazon VisA)
4. **pcb4** (High-Density Board - Amazon VisA)
5. **capsules** (Medical Capsules - Amazon VisA)
6. **bottle** (Bottling & Cap Line - MVTec AD)
7. **cable** (Wiring Harness - MVTec AD)
8. **steel** (Hot-Rolled Strips - NEU Surface Defect)

---
### Pipeline Architecture:
1. **Deep Feature Ingestion & Standardization** (2,048-dimensional ResNet-50 manifold)
2. **Dimensionality Reduction via Principal Component Analysis (PCA)**
3. **K-Means Inertia Elbow Optimization ($K=8$ Manufacturing Clusters)**
4. **Density-Based Defect Isolation (DBSCAN & HDBSCAN)**
5. **2D Manifold Visualizations (t-SNE Projections with Defect Callouts)**
6. **Quantitative Intrinsic & Extrinsic Cluster Validation**"""),

        nbf.v4.new_code_cell("""import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Ensure project root is accessible
sys.path.append("..")

from src.dimensionality_reduction import scale_features, apply_pca, compute_explained_variance, find_optimal_components
from src.clustering import run_kmeans, compute_elbow_inertias, run_dbscan, run_hdbscan, extract_anomaly_indices, get_cluster_summary
from src.visualization import compute_tsne_embedding, plot_tsne_clusters, plot_pca_scree, plot_elbow_curve
from src.evaluation import compute_clustering_metrics

print("✅ All modules loaded successfully!")"""),

        nbf.v4.new_markdown_cell("""### 1. Ingestion of 2,048-Dimensional Deep Feature Embeddings across 8 Lines
We load pre-extracted deep features and metadata indexing 976 items across the 8 industrial categories."""),

        nbf.v4.new_code_cell("""# Load embeddings and metadata
emb_path = Path("../data/embeddings/image_embeddings.npy")
meta_path = Path("../data/embeddings/metadata.csv")

if not emb_path.exists():
    emb_path = Path("../datasets/features.npy")

features = np.load(emb_path)
meta_df = pd.read_csv(meta_path) if meta_path.exists() else None

print(f"📊 Feature Matrix Shape: {features.shape[0]} samples x {features.shape[1]} dimensions")
if meta_df is not None:
    print(f"🏭 Breakdown across 8 Industrial Lines:")
    print(meta_df['category'].value_counts())"""),

        nbf.v4.new_markdown_cell("""### 2. Standardization & Principal Component Analysis (PCA)
Standardize features to zero-mean and unit variance, then decompose variance via PCA retaining 95% of total variance."""),

        nbf.v4.new_code_cell("""# Standardize features
scaled_features, scaler = scale_features(features)

# Retain 95% of total variance
reduced_features, pca_model = apply_pca(scaled_features, explained_variance_ratio=0.95)
ind_var, cum_var = compute_explained_variance(pca_model)

print(f"Reduced from {features.shape[1]} dimensions down to {reduced_features.shape[1]} principal components.")
print(f"Cumulative Explained Variance Retained: {cum_var[-1]*100:.2f}%")

# Generate Scree Plot
os.makedirs("../results/plots", exist_ok=True)
fig_scree = plot_pca_scree(ind_var, cum_var, threshold=0.95, save_path="../results/plots/pca_scree.png")
plt.show()"""),

        nbf.v4.new_markdown_cell(r"""### 3. K-Means Clustering & Inertia Elbow Curve Optimization ($K=2 \dots 12$)
We search for the mathematical inflection point in within-cluster sum-of-squares (WCSS) to discover the natural cluster structure matching our 8 manufacturing categories."""),

        nbf.v4.new_code_cell("""# Compute inertia across cluster counts 2 through 12
k_range = list(range(2, 13))
k_vals, inertias = compute_elbow_inertias(reduced_features, k_range=k_range)

fig_elbow = plot_elbow_curve(k_vals, inertias, save_path="../results/plots/kmeans_elbow.png")
plt.show()

# Fit K-Means with optimal K=8
kmeans_labels, km_model = run_kmeans(reduced_features, n_clusters=8, random_state=42)
km_metrics = compute_clustering_metrics(reduced_features, kmeans_labels)

print("🏆 K-Means Clustering Metrics (K=8):")
for metric_name, val in km_metrics.items():
    print(f"  - {metric_name}: {val}")"""),

        nbf.v4.new_markdown_cell("""### 4. Density-Based Clustering & Unsupervised Defect Discovery (DBSCAN & HDBSCAN)
Density-based clustering partitions the high-density normal component clusters and flags peripheral defective outliers as anomaly noise ($label = -1$)."""),

        nbf.v4.new_code_cell("""# Run DBSCAN
dbscan_labels, _ = run_dbscan(reduced_features, eps=25.0, min_samples=5)
db_summary = get_cluster_summary(dbscan_labels)
db_metrics = compute_clustering_metrics(reduced_features, dbscan_labels, exclude_noise=True)

print("🔍 DBSCAN Discovery Summary:")
print(f"  - Discovered Cohesive Clusters: {db_summary['num_clusters']}")
print(f"  - Isolated Defective Anomalies: {db_summary['num_anomalies']} ({db_summary['anomaly_percentage']:.2f}%)")
print(f"  - Silhouette Score:             {db_metrics.get('silhouette_score', 0):.4f}")

# Run HDBSCAN (if available)
try:
    hdbscan_labels, probs, _ = run_hdbscan(reduced_features, min_cluster_size=5)
    hdb_summary = get_cluster_summary(hdbscan_labels)
    print(f"\\n🔍 HDBSCAN: Found {hdb_summary['num_clusters']} clusters and {hdb_summary['num_anomalies']} anomaly outliers.")
except Exception as e:
    print(f"HDBSCAN note: {e}")"""),

        nbf.v4.new_markdown_cell("""### 5. 2D Manifold Visualizations (t-SNE Projections)
We compute a non-linear 2D t-SNE projection to visualize the separation of the 8 industrial categories and the isolation of defective items."""),

        nbf.v4.new_code_cell("""# Compute 2D t-SNE embedding
tsne_2d = compute_tsne_embedding(reduced_features, n_components=2, perplexity=30.0, random_state=42)

# Plot K-Means Discovered Clusters
fig_km = plot_tsne_clusters(
    tsne_2d,
    kmeans_labels,
    title="t-SNE: Discovered K-Means Manufacturing Clusters (K=8)",
    save_path="../results/plots/tsne_kmeans.png"
)
plt.show()

# Plot DBSCAN Anomaly Discovery
fig_db = plot_tsne_clusters(
    tsne_2d,
    dbscan_labels,
    title="t-SNE: DBSCAN Density Anomaly Discovery (Noise = Anomalies)",
    save_path="../results/plots/tsne_dbscan.png"
)
plt.show()"""),

        nbf.v4.new_markdown_cell("""### 6. Benchmark Summary & Reporting
Save the intrinsic validation metrics across all clustering approaches to `reports/` and `results/metrics/`."""),

        nbf.v4.new_code_cell("""# Compile benchmark table
benchmark_records = [
    {
        "algorithm": "K-Means (K=8)",
        "clusters": km_metrics["evaluated_clusters"],
        "silhouette": round(km_metrics["silhouette_score"], 4),
        "davies_bouldin": round(km_metrics["davies_bouldin_index"], 4),
        "calinski_harabasz": round(km_metrics["calinski_harabasz_index"], 2)
    },
    {
        "algorithm": "DBSCAN (eps=25.0)",
        "clusters": db_metrics["evaluated_clusters"],
        "silhouette": round(db_metrics["silhouette_score"], 4),
        "davies_bouldin": round(db_metrics["davies_bouldin_index"], 4),
        "calinski_harabasz": round(db_metrics["calinski_harabasz_index"], 2)
    }
]

benchmark_df = pd.DataFrame(benchmark_records)
os.makedirs("../reports", exist_ok=True)
benchmark_df.to_csv("../reports/clustering_benchmark.csv", index=False)
print("🏆 Final Clustering Benchmark Scoreboard:")
print(benchmark_df.to_string(index=False))""")
    ]

    nb.cells = cells
    return nb

def build_01_dataset_exploration():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell("""# 01 - Industrial Quality Inspection: Multi-Dataset Exploration
**Authors:** Tanya Tejaswi & Gayatri U

### Objectives:
1. Explore normal reference units and defective parts across the **8-Line Industrial Quality Suite**:
   - **pcb1**, **pcb2**, **pcb3**, **pcb4**, **capsules** (Amazon VisA)
   - **bottle**, **cable** (MVTec AD)
   - **steel** (NEU Surface Defect)
2. Compile structured metadata across 5,800+ raw images.
3. Inspect defect types, label balance, and spatial distributions."""),

        nbf.v4.new_code_cell("""import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append("..")
from src.preprocessing import build_industrial_suite_metadata

print("✅ Loaded preprocessing tools")"""),

        nbf.v4.new_markdown_cell("""### 1. Build & Index Master Industrial Metadata
We scan the dataset directories (`datasets/visa`, `data/mvtec`, `data/steel`) and generate structured metadata."""),

        nbf.v4.new_code_cell("""# Build or load industrial suite metadata
meta_csv = Path("../data/embeddings/metadata_full.csv")
if meta_csv.exists():
    metadata_df = pd.read_csv(meta_csv)
else:
    metadata_df = build_industrial_suite_metadata("..")
    metadata_df.to_csv(meta_csv, index=False)

print(f"📊 Indexed {len(metadata_df)} images across {metadata_df['category'].nunique()} industrial categories.")
summary = pd.crosstab(metadata_df['category'], [metadata_df['split'], metadata_df['is_anomaly']])
print("\\n=== Dataset Split & Defect Distribution ===")
print(summary)"""),

        nbf.v4.new_markdown_cell("""### 2. Visualize Samples across all 8 Manufacturing Lines
Display inspection examples for each line."""),

        nbf.v4.new_code_cell("""categories = ["pcb1", "pcb2", "pcb3", "pcb4", "capsules", "bottle", "cable", "steel"]
fig, axes = plt.subplots(len(categories), 2, figsize=(8, 2.5 * len(categories)), dpi=120)

for idx, cat in enumerate(categories):
    cat_df = metadata_df[metadata_df['category'] == cat]
    good_sample = cat_df[cat_df['is_anomaly'] == 0].iloc[0]
    defect_sample = cat_df[cat_df['is_anomaly'] == 1].iloc[0]
    
    img_good = Image.open(Path("..") / good_sample['image_path']).convert("RGB").resize((180, 180))
    img_defect = Image.open(Path("..") / defect_sample['image_path']).convert("RGB").resize((180, 180))
    
    axes[idx, 0].imshow(img_good)
    axes[idx, 0].set_title(f"{cat.upper()} - Normal", fontsize=9, fontweight='bold', color='#059669')
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(img_defect)
    axes[idx, 1].set_title(f"{cat.upper()} - {defect_sample['defect_type'].upper()}", fontsize=9, fontweight='bold', color='#dc2626')
    axes[idx, 1].axis('off')

plt.tight_layout()
plt.show()""")
    ]

    nb.cells = cells
    return nb

def build_02_feature_extraction_and_anomaly_detection():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell("""# 02 - Feature Extraction & Industrial Anomaly Detection Pipeline
**Author:** Tanya Tejaswi (Computer Vision & Anomaly Detection)

### Complete Workflow:
1. Preprocess images & extract multi-scale spatial patch features.
2. Export global image embeddings (`image_embeddings.npy`) and metadata for Gayathri's PCA, t-SNE, and clustering pipeline.
3. Fit **PatchCore Memory Bank** anomaly detectors strictly on pristine normal units (`train/good`).
4. Evaluate continuous anomaly scoring, ROC-AUC, PR-AUC, F1-scores, and calibrated decision thresholds across all **8 Industrial Categories**:
   - `pcb1`, `pcb2`, `pcb3`, `pcb4`, `capsules`, `bottle`, `cable`, `steel`.
5. Serialize production models to `models/memory_bank_{category}.pkl`."""),

        nbf.v4.new_code_cell("""import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append("..")
from src.anomaly_detection import MemoryBankAnomalyDetector, AnomalyInferenceEngine
from src.evaluation import compute_anomaly_detection_metrics, evaluate_image_level, plot_roc_pr_curves
from src.visualization import plot_anomaly_comparison

print("✅ Anomaly detection modules loaded!")"""),

        nbf.v4.new_markdown_cell("""### 1. Ingest Master Embeddings & Verify Deliverables"""),

        nbf.v4.new_code_cell("""emb_path = Path("../data/embeddings/image_embeddings.npy")
meta_path = Path("../data/embeddings/metadata.csv")

assert emb_path.exists() and meta_path.exists(), "Embeddings deliverable missing!"
features = np.load(emb_path)
meta_df = pd.read_csv(meta_path)

print(f"🎉 Deliverables verified for Gayathri's clustering pipeline:")
print(f"  - Embeddings Array: {features.shape} ({emb_path.stat().st_size / 1e6:.2f} MB)")
print(f"  - Metadata Index:   {len(meta_df)} rows across {meta_df['category'].nunique()} categories")"""),

        nbf.v4.new_markdown_cell("""### 2. Multi-Category Memory Bank Inspection & Benchmark Scoreboard
Load the trained production memory banks (`models/memory_bank_{category}.pkl`) and verify the benchmark scoreboard."""),

        nbf.v4.new_code_cell("""summary_csv = Path("../reports/anomaly_detection_summary.csv")
if summary_csv.exists():
    summary_df = pd.read_csv(summary_csv)
    print("🏆 8-Line Industrial Anomaly Detection Performance Scoreboard:")
    print(summary_df.to_string(index=False))
else:
    print("Run scripts/train_industrial_suite.py to generate summary scoreboard.")"""),

        nbf.v4.new_markdown_cell("""### 3. Model Smoke Test: AnomalyInferenceEngine Deployment"""),

        nbf.v4.new_code_cell("""# Smoke test the inference engine on bottle and pcb1
engine = AnomalyInferenceEngine(category="bottle", models_dir="../models")
test_img = Path("../data/mvtec/bottle/test/broken_large/000.png")

if test_img.exists():
    res = engine.predict(test_img)
    print("✅ Live AnomalyInferenceEngine Prediction Succeeded:")
    print(f"  - Category:    {res['category']}")
    print(f"  - Score:       {res['score']:.4f}")
    print(f"  - Threshold:   {res['threshold']:.4f}")
    print(f"  - Is Defect:   {res['is_defect']}")
    print(f"  - Anomaly Map: {res['anomaly_map'].shape}")
    
    # Display inspection overlay
    plt.figure(figsize=(4, 4), dpi=120)
    plt.imshow(res['heatmap_overlay'])
    plt.title(f"Thermal Heatmap Overlay (Score={res['score']:.1f})", fontsize=10, fontweight='bold')
    plt.axis('off')
    plt.show()""")
    ]

    nb.cells = cells
    return nb

def main():
    nb_dir = PROJECT_ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    # 1. 03_pca_and_clustering.ipynb
    nb03 = build_03_pca_and_clustering()
    nbf.write(nb03, str(nb_dir / "03_pca_and_clustering.ipynb"))
    print("✅ Updated notebooks/03_pca_and_clustering.ipynb")

    # 2. 01_clustering_pipeline_walkthrough.ipynb
    nb01_cl = build_03_pca_and_clustering()
    nbf.write(nb01_cl, str(nb_dir / "01_clustering_pipeline_walkthrough.ipynb"))
    print("✅ Updated notebooks/01_clustering_pipeline_walkthrough.ipynb")

    # 3. 01_dataset_exploration.ipynb
    nb01_ds = build_01_dataset_exploration()
    nbf.write(nb01_ds, str(nb_dir / "01_dataset_exploration.ipynb"))
    print("✅ Updated notebooks/01_dataset_exploration.ipynb")

    # 4. 02_feature_extraction_and_anomaly_detection.ipynb
    nb02 = build_02_feature_extraction_and_anomaly_detection()
    nbf.write(nb02, str(nb_dir / "02_feature_extraction_and_anomaly_detection.ipynb"))
    print("✅ Updated notebooks/02_feature_extraction_and_anomaly_detection.ipynb")

if __name__ == "__main__":
    main()
