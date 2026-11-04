"""
End-to-End Pipeline Runner
Unsupervised Visual Anomaly Discovery for Industrial Quality Inspection
Lead: Gayathri

This script ties together all 4 core modules:
1. Feature Scaling & Dimensionality Reduction (PCA)
2. Unsupervised Clustering (K-Means, DBSCAN, HDBSCAN)
3. 2D Visual Projections (t-SNE) with Anomaly Highlighting
4. Intrinsic Clustering & Benchmark Evaluation
"""

import os
import json
import numpy as np
from sklearn.datasets import make_blobs

# Import core modules
from src.dimensionality_reduction import (
    scale_features,
    apply_pca,
    compute_explained_variance,
    find_optimal_components
)
from src.clustering import (
    run_kmeans,
    compute_elbow_inertias,
    run_dbscan,
    run_hdbscan,
    extract_anomaly_indices,
    get_cluster_summary
)
from src.visualization import (
    compute_tsne_embedding,
    plot_tsne_clusters,
    plot_pca_scree,
    plot_elbow_curve
)
from src.evaluation import compute_clustering_metrics


def load_or_generate_features(data_path: str = "datasets/features.npy"):
    """
    Loads pre-extracted deep features (e.g. from Tanya's ResNet50 extractor),
    or generates a realistic synthetic industrial inspection feature matrix
    (150 normal samples across 3 sub-types + 15 distinct anomaly/defect samples).
    """
    if os.path.exists(data_path):
        print(f"[DATA] Loading real feature embeddings from {data_path}...")
        features = np.load(data_path)
        y_synthetic = None
    else:
        print("[DATA] No pre-extracted 'datasets/features.npy' found.")
        print("[DATA] Generating synthetic industrial inspection embeddings (2,048 dimensions)...")
        np.random.seed(42)
        # 150 normal samples across 3 visual sub-groups (e.g., slight illumination/pose differences)
        X_normal, _ = make_blobs(n_samples=150, centers=3, n_features=2048, cluster_std=3.0, random_state=42)

        # 15 distinct defect anomalies placed outside normal distributions
        outliers = []
        for i in range(15):
            direction = np.random.randn(2048)
            direction = direction / np.linalg.norm(direction)
            outliers.append(direction * np.random.uniform(30.0, 50.0))
        X_anomalies = np.array(outliers)

        features = np.vstack([X_normal, X_anomalies])
        # Ground-truth: 0 for normal, 1 for defect/anomaly (used for validation only)
        y_synthetic = np.array([0] * 150 + [1] * 15)

    return features, y_synthetic


def main():
    print("=" * 80)
    print(" INDUSTRIAL ANOMALY DISCOVERY — END-TO-END PIPELINE")
    print("=" * 80)

    # Ensure output directories exist
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 1: Feature Ingestion & Standardization
    # ------------------------------------------------------------------------
    print("\n--- STEP 1: Feature Ingestion & Standardization ---")
    raw_features, _ = load_or_generate_features()
    n_samples, n_dims = raw_features.shape
    print(f"-> Input feature matrix: {n_samples} samples x {n_dims} dimensions.")

    scaled_features, _ = scale_features(raw_features)
    print("-> Features successfully standardized (zero mean, unit variance).")

    # ------------------------------------------------------------------------
    # STEP 2: Dimensionality Reduction (PCA)
    # ------------------------------------------------------------------------
    print("\n--- STEP 2: Dimensionality Reduction (PCA) ---")
    target_variance = 0.95
    reduced_features, pca_model = apply_pca(scaled_features, explained_variance_ratio=target_variance)
    ind_var, cum_var = compute_explained_variance(pca_model)
    n_components = reduced_features.shape[1]

    print(f"-> Target variance ratio: {target_variance * 100:.1f}%")
    print(f"-> Reduced dimensionality: {n_dims} dims -> {n_components} principal components.")
    print(f"-> Total variance retained: {cum_var[-1] * 100:.2f}%")

    scree_path = "results/plots/pca_scree.png"
    plot_pca_scree(ind_var, cum_var, threshold=target_variance, save_path=scree_path)
    print(f"-> Saved PCA scree plot to: {scree_path}")

    # ------------------------------------------------------------------------
    # STEP 3: K-Means Clustering & Elbow Method
    # ------------------------------------------------------------------------
    print("\n--- STEP 3: Centroid Clustering (K-Means) ---")
    k_range = list(range(2, 9))
    k_vals, inertias = compute_elbow_inertias(reduced_features, k_range=k_range)
    elbow_path = "results/plots/kmeans_elbow.png"
    plot_elbow_curve(k_vals, inertias, save_path=elbow_path)
    print(f"-> Saved K-Means Elbow curve to: {elbow_path}")

    # Run K-Means with k=3
    kmeans_k = 3
    kmeans_labels, _ = run_kmeans(reduced_features, n_clusters=kmeans_k)
    kmeans_metrics = compute_clustering_metrics(reduced_features, kmeans_labels)
    kmeans_summary = get_cluster_summary(kmeans_labels)
    print(f"-> Fitted K-Means (k={kmeans_k}): Silhouette Score = {kmeans_metrics['silhouette_score']:.4f}")

    # ------------------------------------------------------------------------
    # STEP 4: Density Clustering & Anomaly Discovery (DBSCAN)
    # ------------------------------------------------------------------------
    print("\n--- STEP 4: Density Clustering & Anomaly Discovery (DBSCAN) ---")
    dbscan_labels, _ = run_dbscan(reduced_features, eps=25.0, min_samples=5)
    dbscan_summary = get_cluster_summary(dbscan_labels)
    dbscan_anomalies = extract_anomaly_indices(dbscan_labels)
    dbscan_metrics = compute_clustering_metrics(reduced_features, dbscan_labels, exclude_noise=True)

    print(f"-> DBSCAN discovered {dbscan_summary['num_clusters']} normal clusters and "
          f"{dbscan_summary['num_anomalies']} anomalies ({dbscan_summary['anomaly_percentage']:.1f}%).")
    print(f"-> Silhouette Score (excluding noise): {dbscan_metrics['silhouette_score']:.4f}")

    # ------------------------------------------------------------------------
    # STEP 5: Hierarchical Density Clustering (HDBSCAN)
    # ------------------------------------------------------------------------
    print("\n--- STEP 5: Hierarchical Density Clustering (HDBSCAN) ---")
    hdbscan_labels, _, _ = run_hdbscan(reduced_features, min_cluster_size=5)
    hdbscan_summary = get_cluster_summary(hdbscan_labels)
    hdbscan_anomalies = extract_anomaly_indices(hdbscan_labels)
    hdbscan_metrics = compute_clustering_metrics(reduced_features, hdbscan_labels, exclude_noise=True)

    print(f"-> HDBSCAN discovered {hdbscan_summary['num_clusters']} normal clusters and "
          f"{hdbscan_summary['num_anomalies']} anomalies ({hdbscan_summary['anomaly_percentage']:.1f}%).")
    print(f"-> Silhouette Score (excluding noise): {hdbscan_metrics['silhouette_score']:.4f}")

    # ------------------------------------------------------------------------
    # STEP 6: 2D Visual Projections (t-SNE)
    # ------------------------------------------------------------------------
    print("\n--- STEP 6: 2D Visual Projections (t-SNE) ---")
    print("-> Computing 2D t-SNE manifold embedding...")
    tsne_2d = compute_tsne_embedding(reduced_features, n_components=2, perplexity=30.0, max_iter=1000)

    plot_tsne_clusters(
        tsne_2d,
        kmeans_labels,
        title="t-SNE Projection: K-Means Clustering",
        save_path="results/plots/tsne_kmeans.png"
    )
    plot_tsne_clusters(
        tsne_2d,
        dbscan_labels,
        title="t-SNE Projection: DBSCAN Anomaly Discovery",
        save_path="results/plots/tsne_dbscan.png"
    )
    plot_tsne_clusters(
        tsne_2d,
        hdbscan_labels,
        title="t-SNE Projection: HDBSCAN Anomaly Discovery",
        save_path="results/plots/tsne_hdbscan.png"
    )
    print("-> Saved t-SNE projection plots (K-Means, DBSCAN, HDBSCAN) to results/plots/")

    # ------------------------------------------------------------------------
    # STEP 7: Export Comprehensive Benchmark Report
    # ------------------------------------------------------------------------
    print("\n--- STEP 7: Benchmark Summary & Metrics Export ---")
    benchmark_report = {
        "pipeline_metadata": {
            "input_samples": n_samples,
            "raw_dimensions": n_dims,
            "pca_components": n_components,
            "retained_variance_ratio": float(cum_var[-1])
        },
        "clustering_evaluation_metrics": {
            "kmeans": {
                "clusters": kmeans_k,
                "silhouette_score": kmeans_metrics["silhouette_score"],
                "davies_bouldin_index": kmeans_metrics["davies_bouldin_index"],
                "calinski_harabasz_index": kmeans_metrics["calinski_harabasz_index"],
                "summary": kmeans_summary
            },
            "dbscan": {
                "clusters": dbscan_summary["num_clusters"],
                "anomalies_detected": dbscan_summary["num_anomalies"],
                "silhouette_score": dbscan_metrics["silhouette_score"],
                "davies_bouldin_index": dbscan_metrics["davies_bouldin_index"],
                "calinski_harabasz_index": dbscan_metrics["calinski_harabasz_index"],
                "summary": dbscan_summary
            },
            "hdbscan": {
                "clusters": hdbscan_summary["num_clusters"],
                "anomalies_detected": hdbscan_summary["num_anomalies"],
                "silhouette_score": hdbscan_metrics["silhouette_score"],
                "davies_bouldin_index": hdbscan_metrics["davies_bouldin_index"],
                "calinski_harabasz_index": hdbscan_metrics["calinski_harabasz_index"],
                "summary": hdbscan_summary
            }
        }
    }

    metrics_file = "results/metrics/clustering_benchmark.json"
    with open(metrics_file, "w") as f:
        json.dump(benchmark_report, f, indent=4)
    print(f"-> Exported full metrics report to: {metrics_file}")

    print("\n" + "=" * 80)
    print(" CLUSTERING BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Algorithm':<12} | {'Clusters':<9} | {'Anomalies':<10} | {'Silhouette':<11} | {'Davies-Bouldin':<15} | {'Calinski-Harabasz':<18}")
    print("-" * 86)
    print(f"{'K-Means':<12} | {kmeans_summary['num_clusters']:<9} | {'0 (N/A)':<10} | {kmeans_metrics['silhouette_score']:<11.4f} | {kmeans_metrics['davies_bouldin_index']:<15.4f} | {kmeans_metrics['calinski_harabasz_index']:<18.2f}")
    print(f"{'DBSCAN':<12} | {dbscan_summary['num_clusters']:<9} | {dbscan_summary['num_anomalies']:<10} | {dbscan_metrics['silhouette_score']:<11.4f} | {dbscan_metrics['davies_bouldin_index']:<15.4f} | {dbscan_metrics['calinski_harabasz_index']:<18.2f}")
    print(f"{'HDBSCAN':<12} | {hdbscan_summary['num_clusters']:<9} | {hdbscan_summary['num_anomalies']:<10} | {hdbscan_metrics['silhouette_score']:<11.4f} | {hdbscan_metrics['davies_bouldin_index']:<15.4f} | {hdbscan_metrics['calinski_harabasz_index']:<18.2f}")
    print("=" * 80)
    print("Pipeline execution finished successfully!\n")


if __name__ == "__main__":
    main()

