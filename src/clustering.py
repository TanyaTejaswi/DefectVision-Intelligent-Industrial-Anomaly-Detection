"""
Clustering Module
Responsibility: Gayatri U

This module provides unsupervised clustering implementations (K-Means,
DBSCAN, HDBSCAN) to discover groupings and flag anomalous or out-of-distribution
samples in the feature space.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans]:
    """
    Applies K-Means clustering to the feature representation.

    Args:
        features: Input feature matrix of shape (n_samples, n_dimensions).
        n_clusters: Number of clusters (k).
        random_state: Random state seed for reproducible centroid initialization.

    Returns:
        labels: Cluster labels assigned to each sample (0 to n_clusters - 1).
        model: Fitted KMeans model containing cluster centers and inertia.
    """
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")
    if n_clusters > features.shape[0]:
        raise ValueError(f"n_clusters ({n_clusters}) cannot exceed sample count ({features.shape[0]})")

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(features)
    return labels, model


def compute_elbow_inertias(
    features: np.ndarray,
    k_range: Optional[List[int]] = None,
    random_state: int = 42
) -> Tuple[List[int], List[float]]:
    """
    Computes within-cluster sum of squares (inertia) across a range of k values
    to assist with finding the optimal number of clusters using the Elbow method.

    Args:
        features: Input feature matrix of shape (n_samples, n_dimensions).
        k_range: List or range of k values to evaluate (default: 2 to 10).
        random_state: Random state seed for reproducibility.

    Returns:
        k_values: List of evaluated k values.
        inertias: List of corresponding inertia values.
    """
    if k_range is None:
        max_k = min(11, features.shape[0])
        k_range = list(range(2, max_k))

    inertias: List[float] = []
    for k in k_range:
        _, model = run_kmeans(features, n_clusters=k, random_state=random_state)
        inertias.append(float(model.inertia_))

    return list(k_range), inertias


def run_dbscan(
    features: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean"
) -> Tuple[np.ndarray, DBSCAN]:
    """
    Applies Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Identifies core clusters and flags noise/anomalies as -1.

    Args:
        features: Input feature matrix of shape (n_samples, n_dimensions).
        eps: Maximum distance between two samples for one to be in the neighborhood of the other.
        min_samples: Number of samples in a neighborhood for a point to be a core point.
        metric: Distance metric to use ('euclidean', 'cosine', etc.).

    Returns:
        labels: Cluster labels (-1 indicates noise / visual anomaly).
        model: Fitted DBSCAN model.
    """
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be at least 1, got {min_samples}")

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(features)
    return labels, model


def run_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    metric: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray, HDBSCAN]:
    """
    Applies Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN).
    Handles variable-density clusters and produces outlier membership probabilities.

    Args:
        features: Input feature matrix of shape (n_samples, n_dimensions).
        min_cluster_size: Minimum number of samples in a group to be considered a cluster.
        min_samples: Number of samples in a neighborhood for a point to be a core point.
        metric: Metric to use for distance computation.

    Returns:
        labels: Cluster labels for each point (-1 indicates noise / anomaly).
        probabilities: Cluster membership probabilities (0.0 to 1.0).
        model: Fitted HDBSCAN model.
    """
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    if min_cluster_size < 2:
        raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}")

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        copy=True
    )
    labels = model.fit_predict(features)
    probabilities = getattr(model, "probabilities_", np.ones_like(labels, dtype=float))
    return labels, probabilities, model


def extract_anomaly_indices(labels: np.ndarray) -> np.ndarray:
    """
    Extracts the array indices of samples identified as noise/anomalies (label == -1).

    Args:
        labels: Array of cluster labels from DBSCAN or HDBSCAN.

    Returns:
        anomaly_indices: Indices where label is -1.
    """
    return np.where(labels == -1)[0]


def get_cluster_summary(labels: np.ndarray) -> Dict[str, Any]:
    """
    Generates a structured statistical breakdown of cluster assignments.

    Args:
        labels: Array of cluster labels (integers, with -1 denoting noise/anomaly).

    Returns:
        summary: Dictionary containing total samples, cluster count, anomaly count,
                 anomaly percentage, and per-cluster sample counts.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    counts_map = dict(zip(unique_labels.tolist(), counts.tolist()))

    is_noise = (labels == -1)
    num_anomalies = int(np.sum(is_noise))
    total_samples = len(labels)
    normal_clusters = [lbl for lbl in unique_labels if lbl != -1]

    return {
        "total_samples": total_samples,
        "num_clusters": len(normal_clusters),
        "num_anomalies": num_anomalies,
        "anomaly_percentage": float((num_anomalies / total_samples) * 100.0) if total_samples > 0 else 0.0,
        "cluster_distribution": counts_map
    }

