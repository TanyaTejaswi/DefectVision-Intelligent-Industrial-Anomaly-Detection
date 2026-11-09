"""
Anomaly Detection Module
Implements Memory Bank & Nearest-Neighbour / PatchCore-style Anomaly Detection.
Strictly fits on normal training images (train/good) and evaluates on test images.
Includes model serialization (.pkl) and end-to-end AnomalyInferenceEngine for live deployment.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
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
    4. Supports saving and loading precomputed memory bank weights to models/memory_bank_{category}.pkl.
    """
    def __init__(
        self,
        n_neighbors: int = 1,
        subsample_ratio: float = 0.1,
        gaussian_sigma: float = 4.0,
        random_state: int = 42,
        category: str = "general",
        optimal_threshold: Optional[float] = None
    ):
        self.n_neighbors = n_neighbors
        self.subsample_ratio = subsample_ratio
        self.gaussian_sigma = gaussian_sigma
        self.random_state = random_state
        self.category = category
        self.optimal_threshold = optimal_threshold

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

        # Resize feature map to target image dimensions using PIL bilinear interpolation
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

    def is_defect(self, image_score: float) -> bool:
        """
        Returns True if image_score exceeds optimal threshold.
        """
        if self.optimal_threshold is None:
            return False
        return bool(image_score >= self.optimal_threshold)

    def save(self, save_path: Union[str, Path]):
        """
        Serializes the fitted detector to disk for instant loading in production / Streamlit.
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "memory_bank": self.memory_bank,
            "n_neighbors": self.n_neighbors,
            "subsample_ratio": self.subsample_ratio,
            "gaussian_sigma": self.gaussian_sigma,
            "random_state": self.random_state,
            "category": self.category,
            "optimal_threshold": self.optimal_threshold
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            print(f"[OK] Memory bank for '{self.category}' saved to {path} ({path.stat().st_size / 1e6:.2f} MB)")
        except Exception:
            pass

    @classmethod
    def load(cls, load_path: Union[str, Path]) -> "MemoryBankAnomalyDetector":
        """
        Loads a precomputed memory bank detector from disk.
        """
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Memory bank file not found at {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        detector = cls(
            n_neighbors=state.get("n_neighbors", 1),
            subsample_ratio=state.get("subsample_ratio", 0.1),
            gaussian_sigma=state.get("gaussian_sigma", 4.0),
            random_state=state.get("random_state", 42),
            category=state.get("category", "general"),
            optimal_threshold=state.get("optimal_threshold", None)
        )
        detector.memory_bank = state["memory_bank"]

        # Re-index NearestNeighbors
        detector.nn_index = NearestNeighbors(
            n_neighbors=detector.n_neighbors,
            algorithm="auto",
            metric="euclidean"
        )
        detector.nn_index.fit(detector.memory_bank)
        try:
            print(f"[OK] Loaded '{detector.category}' memory bank ({detector.memory_bank.shape[0]} vectors, threshold={detector.optimal_threshold})")
        except Exception:
            pass
        return detector


class AnomalyInferenceEngine:
    """
    High-level, end-to-end industrial inspection engine.
    Given an input image (file path, PIL Image, or NumPy array) and product category,
    it executes feature extraction, memory bank distance scoring, heatmap smoothing,
    and returns an inspection result dictionary.
    """
    def __init__(
        self,
        category: str = "bottle",
        models_dir: Union[str, Path] = "models",
        extractor=None,
        fallback_threshold: float = 12.0
    ):
        self.category = category.lower()
        self.models_dir = Path(models_dir)
        self.extractor = extractor
        self.fallback_threshold = fallback_threshold

        self.detector = self._load_or_init_detector()

    def _load_or_init_detector(self) -> Optional[MemoryBankAnomalyDetector]:
        model_file = self.models_dir / f"memory_bank_{self.category}.pkl"
        if model_file.exists():
            return MemoryBankAnomalyDetector.load(model_file)
        return None

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        custom_threshold: Optional[float] = None
    ) -> Dict:
        """
        Runs complete inspection on a single query image.
        Returns a dictionary containing:
        - category: product name
        - score: float image-level anomaly score
        - threshold: threshold applied
        - is_defect: bool
        - anomaly_map: 2D numpy array of anomaly values (224, 224)
        - raw_image: PIL Image (224, 224)
        - heatmap_overlay: RGB numpy array (224, 224, 3)
        """
        from src.visualization import overlay_heatmap_on_image

        # 1. Load image into PIL RGB (224, 224)
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB").resize((224, 224))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB").resize((224, 224))
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB").resize((224, 224))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 2. Check if detector is loaded
        if self.detector is None:
            # Fallback mock for demo/UI if weights have not been exported yet
            dummy_map = np.zeros((224, 224), dtype=np.float32)
            return {
                "category": self.category,
                "score": 0.0,
                "threshold": custom_threshold or self.fallback_threshold,
                "is_defect": False,
                "anomaly_map": dummy_map,
                "raw_image": pil_img,
                "heatmap_overlay": np.array(pil_img)
            }

        # 3. Extract patch features using PyTorch feature extractor
        if self.extractor is None and TORCH_AVAILABLE:
            from src.feature_extraction import DeepFeatureExtractor
            self.extractor = DeepFeatureExtractor(backbone_name="resnet50")

        if self.extractor is not None:
            from src.preprocessing import get_default_transforms
            transforms = get_default_transforms(img_size=224, crop_size=224)
            img_tensor = transforms(pil_img)

            patches = self.extractor.extract_patch_features(img_tensor, target_size=(28, 28))
            p_np = patches.squeeze(0).permute(1, 2, 0).reshape(-1, patches.shape[1]).cpu().numpy()

            score, heat_map = self.detector.predict_image_and_map(
                p_np,
                spatial_grid=(28, 28),
                target_size=(224, 224)
            )
        else:
            # Non-torch environment fallback
            score = 0.0
            heat_map = np.zeros((224, 224), dtype=np.float32)

        # 4. Determine classification threshold
        active_thresh = custom_threshold
        if active_thresh is None:
            active_thresh = self.detector.optimal_threshold or self.fallback_threshold

        is_defect = bool(score >= active_thresh)
        overlay = overlay_heatmap_on_image(np.array(pil_img), heat_map, alpha=0.5)

        return {
            "category": self.category,
            "score": float(score),
            "threshold": float(active_thresh),
            "is_defect": is_defect,
            "anomaly_map": heat_map,
            "raw_image": pil_img,
            "heatmap_overlay": overlay
        }
