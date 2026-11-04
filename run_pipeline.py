"""
End-to-End Pipeline Runner
Unsupervised Visual Anomaly Discovery for Industrial Quality Inspection
Lead: Gayathri & Tanya

This script ties together all core modules:
1. Feature Scaling & Dimensionality Reduction (PCA)
2. Unsupervised Clustering (K-Means, DBSCAN, HDBSCAN)
3. 2D Visual Projections (t-SNE) with Anomaly Highlighting
4. Intrinsic Clustering Validation (Silhouette, Davies-Bouldin, Calinski-Harabasz)
5. Supervised Anomaly Detection Evaluation (ROC-AUC, PR-AUC, F1-Score)
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
from src.evaluation import (
    compute_clustering_metrics,
    compute_anomaly_detection_metrics
)
from src.anomaly_detection import MemoryBankAnomalyDetector


def load_or_generate_features(data_path: str = "datasets/features.npy"):
    """
    Loads pre-extracted deep features (e.g. from Tanya's ResNet50 extractor),
    or generates a realistic synthetic industrial inspection feature matrix.
    """
    labels_path = "datasets/labels.npy"
    if os.path.exists(data_path):
        print(f"[DATA] Loading real ResNet50 feature embeddings from {data_path}...")
        features = np.load(data_path)
        labels = np.load(labels_path) if os.path.exists(labels_path) else None
    else:
        print("[DATA] No pre-extracted 'datasets/features.npy' found.")
        print("[DATA] Generating synthetic industrial inspection embeddings (2,048 dimensions)...")
        np.random.seed(42)
        X_normal, _ = make_blobs(n_samples=150, centers=3, n_features=2048, cluster_std=3.0, random_state=42)

        outliers = []
        for _ in range(15):
            direction = np.random.randn(2048)
            direction = direction / np.linalg.norm(direction)
            outliers.append(direction * np.random.uniform(30.0, 50.0))
        X_anomalies = np.array(outliers)

        features = np.vstack([X_normal, X_anomalies])
        labels = np.array([0] * 150 + [1] * 15)

    return features, labels


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
    raw_features, ground_truth_labels = load_or_generate_features()
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
    max_k = min(8, n_samples // 2)
    k_range = list(range(2, max(3, max_k + 1)))
    k_vals, inertias = compute_elbow_inertias(reduced_features, k_range=k_range)
    elbow_path = "results/plots/kmeans_elbow.png"
    plot_elbow_curve(k_vals, inertias, save_path=elbow_path)
    print(f"-> Saved K-Means Elbow curve to: {elbow_path}")

    kmeans_k = min(3, n_samples)
    kmeans_labels, _ = run_kmeans(reduced_features, n_clusters=kmeans_k)
    kmeans_metrics = compute_clustering_metrics(reduced_features, kmeans_labels)
    kmeans_summary = get_cluster_summary(kmeans_labels)
    print(f"-> Fitted K-Means (k={kmeans_k}): Silhouette Score = {kmeans_metrics['silhouette_score']:.4f}")

    # ------------------------------------------------------------------------
    # STEP 4: Density Clustering & Anomaly Discovery (DBSCAN)
    # ------------------------------------------------------------------------
    print("\n--- STEP 4: Density Clustering & Anomaly Discovery (DBSCAN) ---")
    # For small sample sizes, adapt eps dynamically if needed
    eps_val = 15.0 if n_samples < 50 else 25.0
    min_samples_val = 3 if n_samples < 50 else 5
    dbscan_labels, _ = run_dbscan(reduced_features, eps=eps_val, min_samples=min_samples_val)
    dbscan_summary = get_cluster_summary(dbscan_labels)
    dbscan_metrics = compute_clustering_metrics(reduced_features, dbscan_labels, exclude_noise=True)

    print(f"-> DBSCAN discovered {dbscan_summary['num_clusters']} normal clusters and "
          f"{dbscan_summary['num_anomalies']} anomalies ({dbscan_summary['anomaly_percentage']:.1f}%).")

    # ------------------------------------------------------------------------
    # STEP 5: Hierarchical Density Clustering (HDBSCAN)
    # ------------------------------------------------------------------------
    print("\n--- STEP 5: Hierarchical Density Clustering (HDBSCAN) ---")
    min_cluster_size_val = 3 if n_samples < 50 else 5
    hdbscan_labels, _, _ = run_hdbscan(reduced_features, min_cluster_size=min_cluster_size_val)
    hdbscan_summary = get_cluster_summary(hdbscan_labels)
    hdbscan_metrics = compute_clustering_metrics(reduced_features, hdbscan_labels, exclude_noise=True)

    print(f"-> HDBSCAN discovered {hdbscan_summary['num_clusters']} normal clusters and "
          f"{hdbscan_summary['num_anomalies']} anomalies ({hdbscan_summary['anomaly_percentage']:.1f}%).")

    # ------------------------------------------------------------------------
    # STEP 6: 2D Visual Projections (t-SNE)
    # ------------------------------------------------------------------------
    print("\n--- STEP 6: 2D Visual Projections (t-SNE) ---")
    effective_perp = min(30.0, float(max(2, (n_samples - 1) // 2)))
    tsne_2d = compute_tsne_embedding(reduced_features, n_components=2, perplexity=effective_perp, max_iter=1000)

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
    # STEP 7: Memory-Bank Anomaly Detector (Supervised Benchmark)
    # ------------------------------------------------------------------------
    anomaly_eval_results = None
    if ground_truth_labels is not None:
        print("\n--- STEP 7: Memory Bank Anomaly Scoring & Evaluation ---")
        normal_idx = np.where(ground_truth_labels == 0)[0]
        if len(normal_idx) > 0 and len(normal_idx) < len(ground_truth_labels):
            detector = MemoryBankAnomalyDetector(n_neighbors=1)
            detector.fit(raw_features[normal_idx])
            anomaly_scores = detector.predict_anomaly_scores(raw_features)
            anomaly_eval_results = compute_anomaly_detection_metrics(ground_truth_labels, anomaly_scores)

            print(f"-> ROC-AUC: {anomaly_eval_results['roc_auc']:.4f}")
            print(f"-> PR-AUC:  {anomaly_eval_results['pr_auc']:.4f}")
            print(f"-> F1-Score: {anomaly_eval_results['f1_score']:.4f} (Precision: {anomaly_eval_results['precision']:.4f}, Recall: {anomaly_eval_results['recall']:.4f})")

    # ------------------------------------------------------------------------
    # STEP 8: Export Comprehensive Benchmark Report
    # ------------------------------------------------------------------------
    print("\n--- STEP 8: Benchmark Summary & Metrics Export ---")
    benchmark_report = {
        "pipeline_metadata": {
            "input_samples": n_samples,
            "raw_dimensions": n_dims,
            "pca_components": n_components,
            "retained_variance_ratio": float(cum_var[-1])
        },
        "clustering_evaluation_metrics (unsupervised)": {
            "kmeans": kmeans_metrics,
            "dbscan": dbscan_metrics,
            "hdbscan": hdbscan_metrics
        },
        "anomaly_detection_metrics (supervised_validation)": anomaly_eval_results
    }

    metrics_file = "results/metrics/clustering_benchmark.json"
    with open(metrics_file, "w") as f:
        json.dump(benchmark_report, f, indent=4)
    print(f"-> Exported full metrics report to: {metrics_file}")

    print("\n" + "=" * 80)
    print(" CLUSTERING & ANOMALY BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Algorithm':<14} | {'Clusters':<9} | {'Anomalies':<10} | {'Silhouette':<12} | {'Davies-Bouldin':<15}")
    print("-" * 75)
    print(f"{'K-Means':<14} | {kmeans_summary['num_clusters']:<9} | {'0 (N/A)':<10} | {kmeans_metrics['silhouette_score']:<12.4f} | {kmeans_metrics['davies_bouldin_index']:<15.4f}")
    db_sil = f"{dbscan_metrics['silhouette_score']:.4f}" if not np.isnan(dbscan_metrics['silhouette_score']) else "N/A"
    db_db = f"{dbscan_metrics['davies_bouldin_index']:.4f}" if not np.isnan(dbscan_metrics['davies_bouldin_index']) else "N/A"
    print(f"{'DBSCAN':<14} | {dbscan_summary['num_clusters']:<9} | {dbscan_summary['num_anomalies']:<10} | {db_sil:<12} | {db_db:<15}")
    hdb_sil = f"{hdbscan_metrics['silhouette_score']:.4f}" if not np.isnan(hdbscan_metrics['silhouette_score']) else "N/A"
    hdb_db = f"{hdbscan_metrics['davies_bouldin_index']:.4f}" if not np.isnan(hdbscan_metrics['davies_bouldin_index']) else "N/A"
    print(f"{'HDBSCAN':<14} | {hdbscan_summary['num_clusters']:<9} | {hdbscan_summary['num_anomalies']:<10} | {hdb_sil:<12} | {hdb_db:<15}")
    print("=" * 80)

    if anomaly_eval_results:
        print(f"Supervised Detection Performance: ROC-AUC = {anomaly_eval_results['roc_auc']:.2%} | PR-AUC = {anomaly_eval_results['pr_auc']:.2%} | F1 = {anomaly_eval_results['f1_score']:.2%}")
        print("=" * 80)

    print("Pipeline execution finished successfully!\n")


if __name__ == "__main__":
    main()
