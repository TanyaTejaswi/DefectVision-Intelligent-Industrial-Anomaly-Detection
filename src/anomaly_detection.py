"""
Anomaly Detection Module
Implements Memory Bank & Nearest-Neighbour / PatchCore-style Anomaly Detection.
Strictly fits on normal training images (train/good) and evaluates on test images.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MemoryBankAnomalyDetector:
    """
    PatchCore-style memory bank anomaly detector.

    Key principles:
    1. Memory bank is populated ONLY with patch features from pristine 'train/good' images.
    2. Test image anomaly score is calculated using nearest-neighbour distance to the memory bank.
    3. Patch anomaly scores are reshaped to spatial grid and upsampled to form an anomaly heatmap.
    """
    def __init__(
        self,
        n_neighbors: int = 1,
        subsample_ratio: float = 0.1,
        gaussian_sigma: float = 4.0,
        random_state: int = 42
    ):
        self.n_neighbors = n_neighbors
        self.subsample_ratio = subsample_ratio
        self.gaussian_sigma = gaussian_sigma
        self.random_state = random_state

        self.memory_bank: Optional[np.ndarray] = None
        self.nn_index: Optional[NearestNeighbors] = None

    def fit(self, normal_patch_features: np.ndarray):
        """
        Builds the memory bank from normal training patches.
        Args:
            normal_patch_features: shape (N_patches, Feature_dim)
        """
        rng = np.random.RandomState(self.random_state)
        total_patches = normal_patch_features.shape[0]

        # Coreset subsampling (or random subsampling for efficiency)
        if self.subsample_ratio < 1.0:
            sample_size = max(1, int(total_patches * self.subsample_ratio))
            indices = rng.choice(total_patches, size=sample_size, replace=False)
            self.memory_bank = normal_patch_features[indices]
        else:
            self.memory_bank = normal_patch_features

        # Fit Scikit-Learn NearestNeighbors index
        self.nn_index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="auto",
            metric="euclidean"
        )
        self.nn_index.fit(self.memory_bank)
        print(f"Memory bank built with {self.memory_bank.shape[0]} normal patch vectors (dim={self.memory_bank.shape[1]}).")

    def predict_image_and_map(
        self,
        patch_features: np.ndarray,
        spatial_grid: Tuple[int, int] = (28, 28),
        target_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[float, np.ndarray]:
        """
        Calculates anomaly score and 2D anomaly heatmap for a single test image.
        Args:
            patch_features: shape (H*W, feature_dim)
            spatial_grid: (H_feat, W_feat), e.g. (28, 28)
            target_size: original image dimensions (H, W), e.g. (224, 224)
        Returns:
            image_score: float (maximum patch distance)
            anomaly_map: shape target_size (smoothed heatmap)
        """
        if self.nn_index is None:
            raise RuntimeError("MemoryBankAnomalyDetector is not fitted yet. Call fit() first.")

        # Compute nearest-neighbour distance to normal memory bank
        distances, _ = self.nn_index.kneighbors(patch_features)
        patch_scores = distances[:, 0]  # distance to nearest normal neighbor

        # Maximum patch distance is the image-level anomaly score
        image_score = float(np.max(patch_scores))

        # Reshape to 2D feature map
        h_grid, w_grid = spatial_grid
        feat_map = patch_scores.reshape((h_grid, w_grid))

        # Resize feature map to target image dimensions using PIL or SciPy
        from PIL import Image
        feat_img = Image.fromarray(feat_map)
        resized_map = feat_img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
        anomaly_map = np.array(resized_map, dtype=np.float32)

        # Smooth using Gaussian filter
        if self.gaussian_sigma > 0:
            anomaly_map = gaussian_filter(anomaly_map, sigma=self.gaussian_sigma)

        return image_score, anomaly_map

    def predict_batch_scores(self, image_patch_list: List[np.ndarray]) -> np.ndarray:
        """
        Computes image-level anomaly scores for a list of test images.
        """
        scores = []
        for patches in image_patch_list:
            score, _ = self.predict_image_and_map(patches)
            scores.append(score)
        return np.array(scores)
