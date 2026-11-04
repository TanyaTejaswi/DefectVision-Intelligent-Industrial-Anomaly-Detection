"""
Unit tests for Clustering module (src/clustering.py)
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from src.clustering import (
    run_kmeans,
    compute_elbow_inertias,
    run_dbscan,
    run_hdbscan,
    extract_anomaly_indices,
    get_cluster_summary
)


@pytest.fixture
def synthetic_clusters_with_outliers():
    """
    Creates 3 tight clusters with 90 normal points plus 5 far outliers.
    Simulates normal product clusters and distinct defect/noise samples.
    """
    X, _ = make_blobs(n_samples=90, centers=3, n_features=10, cluster_std=0.5, random_state=42)
    # 5 isolated distant outlier points (each at a distinct location)
    outliers = np.array([
        [50.0] * 10,
        [-50.0] * 10,
        [80.0, -80.0] * 5,
        [-40.0, 60.0] * 5,
        [70.0, 30.0] * 5
    ])
    return np.vstack([X, outliers])


def test_run_kmeans(synthetic_clusters_with_outliers):
    X = synthetic_clusters_with_outliers
    labels, model = run_kmeans(X, n_clusters=3, random_state=42)

    assert labels.shape == (len(X),)
    assert len(np.unique(labels)) == 3
    assert model.cluster_centers_.shape == (3, 10)
    assert model.inertia_ > 0.0


def test_compute_elbow_inertias(synthetic_clusters_with_outliers):
    X = synthetic_clusters_with_outliers
    k_range = [2, 3, 4, 5]
    k_vals, inertias = compute_elbow_inertias(X, k_range=k_range, random_state=42)

    assert k_vals == k_range
    assert len(inertias) == len(k_range)
    # Inertia must decrease or stay equal as k increases
    for i in range(len(inertias) - 1):
        assert inertias[i] >= inertias[i + 1]


def test_run_dbscan(synthetic_clusters_with_outliers):
    X = synthetic_clusters_with_outliers
    # Dense clusters will be found; far outliers should be labeled -1
    labels, model = run_dbscan(X, eps=2.0, min_samples=5)

    assert labels.shape == (len(X),)
    # The 5 outliers at index 90..94 should be detected as noise (-1)
    anomalies = extract_anomaly_indices(labels)
    assert len(anomalies) >= 5
    for idx in range(90, 95):
        assert idx in anomalies


def test_run_hdbscan(synthetic_clusters_with_outliers):
    X = synthetic_clusters_with_outliers
    labels, probabilities, model = run_hdbscan(X, min_cluster_size=5)

    assert labels.shape == (len(X),)
    assert probabilities.shape == (len(X),)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    # Far outliers should be labeled -1
    anomalies = extract_anomaly_indices(labels)
    assert len(anomalies) >= 5


def test_extract_anomaly_indices():
    sample_labels = np.array([0, 1, -1, 1, 0, -1, 2])
    anomalies = extract_anomaly_indices(sample_labels)
    np.testing.assert_array_equal(anomalies, np.array([2, 5]))


def test_get_cluster_summary():
    sample_labels = np.array([0, 0, 1, 1, 1, -1])
    summary = get_cluster_summary(sample_labels)

    assert summary["total_samples"] == 6
    assert summary["num_clusters"] == 2
    assert summary["num_anomalies"] == 1
    assert pytest.approx(summary["anomaly_percentage"], 0.01) == 16.67
    assert summary["cluster_distribution"][0] == 2
    assert summary["cluster_distribution"][1] == 3
    assert summary["cluster_distribution"][-1] == 1

