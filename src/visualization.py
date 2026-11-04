"""
Visualization Module
Generates anomaly heatmaps, overlay blends, side-by-side defect comparisons,
and ROC / PR evaluation curves.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    """Normalizes array to range [0, 1]."""
    min_val = np.min(heatmap)
    max_val = np.max(heatmap)
    if max_val - min_val > 1e-8:
        return (heatmap - min_val) / (max_val - min_val)
    return np.zeros_like(heatmap)


def overlay_heatmap_on_image(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet"
) -> np.ndarray:
    """
    Overlays a continuous anomaly heatmap onto an RGB image.
    Args:
        image: RGB numpy array (H, W, 3) in [0, 255] or PIL Image
        heatmap: 2D numpy array (H, W)
        alpha: image blending weight (1-alpha is heatmap weight)
        cmap: matplotlib colormap name ('jet', 'plasma', 'inferno')
    Returns:
        blended: RGB uint8 numpy array (H, W, 3)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Normalize heatmap
    norm_map = normalize_map(heatmap)

    # Apply colormap
    colormap = plt.get_cmap(cmap)
    colored_map = colormap(norm_map)[:, :, :3]  # (H, W, 3) in [0, 1]
    colored_map = (colored_map * 255).astype(np.uint8)

    # Resize image if dimensions mismatch
    if image.shape[:2] != heatmap.shape[:2]:
        img_pil = Image.fromarray(image).resize((heatmap.shape[1], heatmap.shape[0]))
        image = np.array(img_pil)

    # Alpha blend
    blended = (alpha * image + (1.0 - alpha) * colored_map).astype(np.uint8)
    return blended


def plot_anomaly_comparison(
    image: Union[np.ndarray, Image.Image],
    ground_truth_mask: Optional[np.ndarray],
    anomaly_map: np.ndarray,
    anomaly_score: float,
    defect_type: str = "defect",
    category: str = "bottle",
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plots a multi-panel visual inspection:
    [Input Image | Ground Truth Mask | Predicted Anomaly Map | Anomaly Overlay]
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    overlay = overlay_heatmap_on_image(image, anomaly_map)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # 1. Original Image
    axes[0].imshow(image)
    axes[0].set_title(f"Input: {category} ({defect_type})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth Mask
    if ground_truth_mask is not None and np.sum(ground_truth_mask) > 0:
        axes[1].imshow(ground_truth_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
    else:
        axes[1].imshow(np.zeros_like(anomaly_map), cmap="gray")
        axes[1].set_title("Ground Truth: Normal (No Defect)", fontsize=11)
    axes[1].axis("off")

    # 3. Anomaly Map
    im = axes[2].imshow(anomaly_map, cmap="jet")
    axes[2].set_title(f"Anomaly Map (Score: {anomaly_score:.3f})", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Anomaly Overlay
    axes[3].imshow(overlay)
    axes[3].set_title("Inspection Overlay", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")

    return fig
