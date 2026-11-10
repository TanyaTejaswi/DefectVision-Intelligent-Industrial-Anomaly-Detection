"""
Anomaly Detection Module

Implements memory-bank and PatchCore-style nearest-neighbour
anomaly detection for industrial quality inspection.

Features:
1. Builds a memory bank using normal (defect-free) features.
2. Calculates anomaly scores using nearest-neighbour distances.
3. Predicts normal or anomalous samples using a threshold.
4. Generates spatial defect localization heatmaps.
5. Overlays heatmaps onto original inspection images.
6. Supports image-level and batch-level anomaly prediction.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, zoom
from sklearn.neighbors import NearestNeighbors
from PIL import Image


class MemoryBankAnomalyDetector:
    """
    PatchCore-style nearest-neighbour memory bank detector.

    The memory bank is built using only normal training features.
    Test features are compared against this bank to calculate
    anomaly scores and identify potential defects.
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        metric: str = "euclidean",
        subsample_ratio: float = 1.0,
        gaussian_sigma: float = 4.0,
        random_state: int = 42
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.subsample_ratio = subsample_ratio
        self.gaussian_sigma = gaussian_sigma
        self.random_state = random_state

        self.memory_bank: Optional[np.ndarray] = None
        self.nn_model: Optional[NearestNeighbors] = None

        # Alias retained for compatibility with Tanya's implementation.
        self.nn_index: Optional[NearestNeighbors] = None

        self.threshold: Optional[float] = None

    def fit(self, normal_features: np.ndarray) -> "MemoryBankAnomalyDetector":
        """
        Builds the memory bank using normal training features.

        Args:
            normal_features: Array of shape (N, feature_dimensions).

        Returns:
            Fitted detector.
        """

        if normal_features.ndim != 2:
            raise ValueError("Features must be a 2D array.")

        n_samples = normal_features.shape[0]

        if n_samples == 0:
            raise ValueError("Cannot fit memory bank with zero samples.")

        if not 0 < self.subsample_ratio <= 1:
            raise ValueError("subsample_ratio must be greater than 0 and at most 1.")

        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")

        # Reproducible memory-bank subsampling
        if self.subsample_ratio < 1.0:
            rng = np.random.RandomState(self.random_state)

            sample_count = max(
                1,
                int(n_samples * self.subsample_ratio)
            )

            indices = rng.choice(
                n_samples,
                size=sample_count,
                replace=False
            )

            self.memory_bank = normal_features[indices].copy()

        else:
            self.memory_bank = normal_features.copy()

        # Fit nearest-neighbour model
        k = min(self.n_neighbors, self.memory_bank.shape[0])

        self.nn_model = NearestNeighbors(
            n_neighbors=k,
            metric=self.metric,
            algorithm="auto"
        )

        self.nn_model.fit(self.memory_bank)

        # Compatibility alias
        self.nn_index = self.nn_model

        # Calculate training baseline threshold
        distances, _ = self.nn_model.kneighbors(
            self.memory_bank,
            n_neighbors=k
        )

        # Exclude self-distance when possible
        if k > 1:
            training_distances = np.mean(distances[:, 1:], axis=1)
        else:
            training_distances = distances[:, 0]

        self.threshold = float(
            np.percentile(training_distances, 95)
        )

        print(
            f"Memory bank built with {self.memory_bank.shape[0]} "
            f"normal feature vectors."
        )

        return self

    def predict_anomaly_scores(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Calculates anomaly scores for test features.

        Higher scores indicate greater deviation from normal patterns.
        """

        if self.nn_model is None:
            raise RuntimeError("Call fit() before prediction.")

        if features.ndim != 2:
            raise ValueError("Features must be a 2D array.")

        distances, _ = self.nn_model.kneighbors(features)

        return distances[:, -1]

    def predict(
        self,
        features: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predicts binary labels.

        0 = Normal
        1 = Anomaly
        """

        scores = self.predict_anomaly_scores(features)

        cutoff = self.threshold if threshold is None else threshold

        if cutoff is None:
            cutoff = float(np.median(scores))

        return (scores >= cutoff).astype(int)

    def predict_image_and_map(
        self,
        patch_features: np.ndarray,
        spatial_grid: Tuple[int, int] = (28, 28),
        target_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[float, np.ndarray]:
        """
        Calculates an image-level anomaly score and localization map.

        Args:
            patch_features: Array of shape (H*W, feature_dimensions).
            spatial_grid: Height and width of the patch feature grid.
            target_size: Output heatmap dimensions (H, W).

        Returns:
            image_score: Maximum patch-level anomaly score.
            anomaly_map: Upsampled spatial anomaly heatmap.
        """

        if self.nn_model is None:
            raise RuntimeError("Call fit() before prediction.")

        distances, _ = self.nn_model.kneighbors(patch_features)

        patch_scores = distances[:, -1]

        image_score = float(np.max(patch_scores))

        expected_patches = spatial_grid[0] * spatial_grid[1]

        if len(patch_scores) != expected_patches:
            raise ValueError(
                f"Expected {expected_patches} patches, "
                f"but received {len(patch_scores)}."
            )

        anomaly_map = generate_defect_heatmap(
            patch_scores,
            grid_size=spatial_grid,
            target_size=target_size,
            sigma=self.gaussian_sigma
        )

        return image_score, anomaly_map

    def predict_batch_scores(
        self,
        image_patch_list: list
    ) -> np.ndarray:
        """
        Calculates image-level anomaly scores for multiple images.

        Each element contains the patch features of one image.
        """

        scores = []

        for patches in image_patch_list:
            image_score, _ = self.predict_image_and_map(patches)
            scores.append(image_score)

        return np.array(scores)


def generate_defect_heatmap(
    patch_scores: np.ndarray,
    grid_size: Tuple[int, int] = (14, 14),
    target_size: Tuple[int, int] = (224, 224),
    sigma: float = 2.0
) -> np.ndarray:
    """
    Converts patch-level anomaly scores into a smooth 2D heatmap.
    """

    total_patches = grid_size[0] * grid_size[1]

    if len(patch_scores) != total_patches:
        raise ValueError(
            f"Expected {total_patches} patch scores, "
            f"but received {len(patch_scores)}."
        )

    grid_2d = patch_scores.reshape(grid_size)

    smoothed_grid = gaussian_filter(
        grid_2d,
        sigma=sigma
    )

    zoom_y = target_size[0] / grid_size[0]
    zoom_x = target_size[1] / grid_size[1]

    heatmap = zoom(
        smoothed_grid,
        (zoom_y, zoom_x),
        order=1
    )

    min_val = np.min(heatmap)
    max_val = np.max(heatmap)

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
    Overlays a colored anomaly heatmap onto an RGB image.

    Returns an RGB image in the range [0, 255].
    """

    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image

    if img_uint8.shape[:2] != heatmap.shape:
        raise ValueError(
            "Image and heatmap must have matching height and width."
        )

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    cmap = plt.get_cmap(colormap)

    colored_heatmap = cmap(heatmap)[:, :, :3]

    colored_heatmap_uint8 = (
        colored_heatmap * 255
    ).astype(np.uint8)

    blended = (
        (1.0 - alpha) * img_uint8
        + alpha * colored_heatmap_uint8
    )

    return np.clip(blended, 0, 255).astype(np.uint8)
