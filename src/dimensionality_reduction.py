"""
Dimensionality Reduction Module
Responsibility: Gayathri

This module handles projection and dimensionality reduction of high-dimensional
feature embeddings (e.g., ResNet50 features), including Principal Component
Analysis (PCA) and cumulative explained variance analysis.
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def scale_features(features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardizes features by removing the mean and scaling to unit variance.

    Args:
        features: Input feature matrix of shape (n_samples, n_features).

    Returns:
        scaled_features: Zero-mean, unit-variance standardized features.
        scaler: Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


def apply_pca(
    features: np.ndarray,
    n_components: Optional[int] = None,
    explained_variance_ratio: Optional[float] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """
    Fits and applies Principal Component Analysis (PCA) to high-dimensional features.

    Args:
        features: Input feature matrix of shape (n_samples, n_features).
        n_components: Exact number of principal components to keep (integer).
        explained_variance_ratio: Target variance ratio (0.0 to 1.0, e.g., 0.95 to retain 95%).
        random_state: Random state for reproducibility.

    Returns:
        reduced_features: Transformed feature array of shape (n_samples, n_components).
        pca_model: Fitted PCA model object.
    """
    if features.ndim != 2:
        raise ValueError(f"Features array must be 2D of shape (n_samples, n_features), got shape {features.shape}")

    # Determine component selection mode
    if explained_variance_ratio is not None:
        if not (0.0 < explained_variance_ratio <= 1.0):
            raise ValueError(f"explained_variance_ratio must be between 0.0 and 1.0, got {explained_variance_ratio}")
        pca = PCA(n_components=explained_variance_ratio, svd_solver="full", random_state=random_state)
    elif n_components is not None:
        if n_components <= 0:
            raise ValueError(f"n_components must be a positive integer, got {n_components}")
        pca = PCA(n_components=n_components, random_state=random_state)
    else:
        # Default: keep all components up to min(n_samples, n_features)
        pca = PCA(random_state=random_state)

    reduced_features = pca.fit_transform(features)
    return reduced_features, pca


def compute_explained_variance(pca_model: PCA) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts individual and cumulative explained variance ratios from a fitted PCA model.

    Args:
        pca_model: A fitted PCA instance.

    Returns:
        individual_variance: Ratio of variance explained by each principal component.
        cumulative_variance: Cumulative sum of explained variance ratios.
    """
    if not hasattr(pca_model, "explained_variance_ratio_"):
        raise ValueError("The provided PCA model has not been fitted yet.")

    individual_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(individual_variance)
    return individual_variance, cumulative_variance


def find_optimal_components(cumulative_variance: np.ndarray, threshold: float = 0.95) -> int:
    """
    Finds the minimum number of principal components required to retain
    at least the specified cumulative variance threshold.

    Args:
        cumulative_variance: Cumulative variance array from compute_explained_variance.
        threshold: Desired explained variance threshold (default 0.95 = 95%).

    Returns:
        n_components: Minimum number of components required (1-indexed).
    """
    indices = np.where(cumulative_variance >= threshold)[0]
    if len(indices) == 0:
        return len(cumulative_variance)
    return int(indices[0] + 1)
