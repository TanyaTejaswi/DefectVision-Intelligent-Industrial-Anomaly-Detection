"""
Unit tests for Evaluation module (src/evaluation.py)
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from src.evaluation import (
    compute_clustering_metrics,
    compute_anomaly_detection_metrics,
    compute_posthoc_cluster_alignment
)


@pytest.fixture
def well_separated_clusters():
    X, y = make_blobs(n_samples=90, centers=3, n_features=5, cluster_std=0.2, random_state=42)
    return X, y


def test_compute_clustering_metrics(well_separated_clusters):
    X, labels = well_separated_clusters
    metrics = compute_clustering_metrics(X, labels)

    assert "silhouette_score" in metrics
    assert "davies_bouldin_index" in metrics
    assert "calinski_harabasz_index" in metrics

    # Tightly separated clusters should achieve high silhouette (> 0.5) and low Davies-Bouldin (< 1.0)
    assert metrics["silhouette_score"] > 0.5
    assert metrics["davies_bouldin_index"] < 1.0
    assert metrics["calinski_harabasz_index"] > 50.0
    assert metrics["evaluated_samples"] == 90
    assert metrics["evaluated_clusters"] == 3


def test_compute_clustering_metrics_with_noise(well_separated_clusters):
    X, labels = well_separated_clusters
    # Introduce 5 noise points labeled -1
    labels_with_noise = labels.copy()
    labels_with_noise[:5] = -1

    metrics = compute_clustering_metrics(X, labels_with_noise, exclude_noise=True)
    # 90 total minus 5 noise points = 85 evaluated
    assert metrics["evaluated_samples"] == 85
    assert not np.isnan(metrics["silhouette_score"])


def test_compute_clustering_metrics_single_cluster():
    X = np.random.randn(20, 5)
    labels = np.zeros(20, dtype=int)  # Only 1 cluster
    metrics = compute_clustering_metrics(X, labels)

    # Cannot compute silhouette with only 1 cluster; should gracefully return NaN
    assert np.isnan(metrics["silhouette_score"])
    assert np.isnan(metrics["davies_bouldin_index"])


def test_compute_anomaly_detection_metrics():
    np.random.seed(42)
    # 80 normal items (label 0) with low anomaly scores [0.0, 0.4]
    scores_normal = np.random.uniform(0.0, 0.4, size=80)
    # 20 defective items (label 1) with high anomaly scores [0.6, 1.0]
    scores_defect = np.random.uniform(0.6, 1.0, size=20)

    y_true = np.array([0] * 80 + [1] * 20)
    scores = np.concatenate([scores_normal, scores_defect])

    metrics = compute_anomaly_detection_metrics(y_true, scores)

    assert metrics["roc_auc"] > 0.95
    assert metrics["pr_auc"] > 0.90
    assert metrics["precision"] > 0.90
    assert metrics["recall"] > 0.90
    assert metrics["f1_score"] > 0.90


def test_compute_posthoc_cluster_alignment():
    labels_true = np.array([0, 0, 1, 1, 2, 2])
    # Exact match
    metrics = compute_posthoc_cluster_alignment(labels_true, labels_true)
    assert pytest.approx(metrics["adjusted_rand_index"], 0.001) == 1.0
    assert pytest.approx(metrics["normalized_mutual_info"], 0.001) == 1.0

    # Permuted labels (cluster naming shouldn't matter to ARI / NMI)
    labels_perm = np.array([2, 2, 0, 0, 1, 1])
    perm_metrics = compute_posthoc_cluster_alignment(labels_true, labels_perm)
    assert pytest.approx(perm_metrics["adjusted_rand_index"], 0.001) == 1.0
    assert pytest.approx(perm_metrics["normalized_mutual_info"], 0.001) == 1.0

