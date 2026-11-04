"""
Anomaly Detection Module
Responsibility: Tanya (Computer Vision & Anomaly Detection Lead)

This module implements:
1. Memory-Bank Anomaly Detector:
   - Stores feature embeddings from normal (defect-free) training images.
   - Computes k-Nearest Neighbor distances for test images / patches.
2. Anomaly Scoring:
   - Derives continuous anomaly scores based on distance to nearest normal pattern.
3. Defect Localization Heatmaps:
   - Converts patch-level distances into spatial anomaly heatmaps using Gaussian smoothing.
   - Blends defect heatmaps with original industrial inspection photos.
"""

from typing import Tuple, Optional, Dict, Any, List
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class MemoryBankAnomalyDetector:
    """
    Nearest-Neighbor Memory Bank Anomaly Detector for industrial quality inspection.
    Learns the distribution of defect-free products by building a reference bank
    of normal embeddings and scoring test samples based on their Euclidean distance
    to the closest normal neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        metric: str = "euclidean",
        subsample_ratio: float = 1.0,
        random_state: int = 42
    ):
        """
        Args:
            n_neighbors: Number of nearest neighbors to query (default: 1).
            metric: Distance metric ('euclidean', 'cosine', etc.).
            subsample_ratio: Fraction of normal features to retain in memory bank (0.0 to 1.0).
            random_state: Random state for reproducible subsampling.
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state

        self.memory_bank: Optional[np.ndarray] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self.threshold: Optional[float] = None

    def fit(self, normal_features: np.ndarray) -> "MemoryBankAnomalyDetector":
        """
        Builds the memory bank using only normal (good) product features.

        Args:
            normal_features: Feature matrix of shape (n_normal_samples, n_dimensions).

        Returns:
            self: Fitted detector instance.
        """
        if normal_features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got shape {normal_features.shape}")

        n_samples = normal_features.shape[0]
        if n_samples == 0:
            raise ValueError("Cannot fit memory bank with 0 samples.")

        # Subsample memory bank if requested (useful for large patch banks)
        if 0.0 < self.subsample_ratio < 1.0:
            np.random.seed(self.random_state)
            sample_count = max(1, int(n_samples * self.subsample_ratio))
            indices = np.random.choice(n_samples, size=sample_count, replace=False)
            self.memory_bank = normal_features[indices]
        else:
            self.memory_bank = np.copy(normal_features)

        # Fit nearest neighbors model on normal memory bank
        k = min(self.n_neighbors, self.memory_bank.shape[0])
        self.nn_model = NearestNeighbors(n_neighbors=k, metric=self.metric, algorithm="auto")
        self.nn_model.fit(self.memory_bank)

        # Compute training baseline distances to set default threshold (95th percentile)
        distances, _ = self.nn_model.kneighbors(self.memory_bank)
        mean_distances = np.mean(distances, axis=1)
        self.threshold = float(np.percentile(mean_distances, 95.0))

        return self

    def predict_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Computes anomaly scores for test samples based on distance to nearest normal neighbor.

        Args:
            features: Test feature matrix of shape (n_test_samples, n_dimensions).

        Returns:
            scores: Continuous anomaly scores of shape (n_test_samples,).
        """
        if self.nn_model is None or self.memory_bank is None:
            raise ValueError("The detector has not been fitted yet. Call fit() first.")
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got shape {features.shape}")

        distances, _ = self.nn_model.kneighbors(features)
        # Average distance across the k nearest normal neighbors
        scores = np.mean(distances, axis=1)
        return scores

    def predict(self, features: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predicts binary labels: 0 for Normal, 1 for Anomaly/Defect.

        Args:
            features: Test feature matrix of shape (n_test_samples, n_dimensions).
            threshold: Cutoff score. If None, uses the 95th percentile training threshold.

        Returns:
            binary_predictions: Array of shape (n_test_samples,) with 0s and 1s.
        """
        scores = self.predict_anomaly_scores(features)
        cutoff = self.threshold if threshold is None else threshold
        if cutoff is None:
            cutoff = float(np.median(scores))
        return (scores >= cutoff).astype(int)


def generate_defect_heatmap(
    patch_scores: np.ndarray,
    grid_size: Tuple[int, int] = (14, 14),
    target_size: Tuple[int, int] = (224, 224),
    sigma: float = 2.0
) -> np.ndarray:
    """
    Converts 1D patch-level anomaly scores into a smooth 2D defect localization heatmap.

    Args:
        patch_scores: Array of anomaly scores for each spatial patch (e.g. 196 scores for 14x14).
        grid_size: Spatial grid dimensions (H_grid, W_grid) matching patch layout.
        target_size: Target image dimensions (H_img, W_img).
        sigma: Standard deviation for Gaussian smoothing filter.

    Returns:
        heatmap: Normalized 2D anomaly heatmap of shape (H_img, W_img) with values in [0.0, 1.0].
    """
    total_patches = grid_size[0] * grid_size[1]
    if len(patch_scores) != total_patches:
        raise ValueError(f"Expected {total_patches} patch scores for grid {grid_size}, got {len(patch_scores)}")

    # Reshape to 2D patch grid
    grid_2d = patch_scores.reshape(grid_size)

    # Apply Gaussian smoothing to blend discrete patch borders
    smoothed_grid = gaussian_filter(grid_2d, sigma=sigma)

    # Resize/interpolate to full image dimensions using bilinear interpolation
    zoom_y = target_size[0] / grid_size[0]
    zoom_x = target_size[1] / grid_size[1]
    from scipy.ndimage import zoom
    heatmap = zoom(smoothed_grid, (zoom_y, zoom_x), order=1)

    # Normalize heatmap between 0.0 and 1.0
    min_val, max_val = np.min(heatmap), np.max(heatmap)
    if max_val > min_val:
        heatmap = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Overlays a colored defect heatmap onto an inspection image.

    Args:
        image: Original image as RGB numpy array of shape (H, W, 3), range [0, 255] or [0, 1].
        heatmap: 2D anomaly heatmap of shape (H, W), range [0.0, 1.0].
        alpha: Heatmap transparency factor (0.0 = image only, 1.0 = heatmap only).
        colormap: Matplotlib colormap name ('jet', 'inferno', 'plasma').

    Returns:
        blended: RGB image array of shape (H, W, 3) with heatmap overlay in range [0, 255].
    """
    # Normalize image to uint8 [0, 255]
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image

    # Apply colormap to heatmap
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Drop alpha channel -> RGB in [0, 1]
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

    # Alpha blend
    blended = (1.0 - alpha) * img_uint8 + alpha * colored_heatmap_uint8
    return np.clip(blended, 0, 255).astype(np.uint8)

