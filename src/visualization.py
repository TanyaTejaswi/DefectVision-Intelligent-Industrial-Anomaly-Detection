"""
Visualization Module
Responsibility: Gayathri

This module handles 2D visual projections (t-SNE), PCA scree plots,
elbow curves, and cluster distribution diagrams for visual anomaly discovery.
"""

from pathlib import Path
from typing import List, Optional, Union
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

def compute_tsne_embedding(
    features: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> np.ndarray:
    """
    Computes t-Distributed Stochastic Neighbor Embedding (t-SNE) for high-dimensional data.

    Args:
        features: Input feature matrix of shape (n_samples, n_features).
        n_components: Dimension of the embedded space (typically 2).
        perplexity: Neighborhood balance parameter. Automatically capped if n_samples <= perplexity.
        max_iter: Maximum number of optimization iterations.
        random_state: Seed for random number generator.

    Returns:
        embedding: Low-dimensional projection coordinates of shape (n_samples, n_components).
    """
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")

    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for t-SNE, got {n_samples}")

    # Perplexity must be strictly less than n_samples
    effective_perplexity = min(perplexity, float(max(1, n_samples - 1)))

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        max_iter=max_iter,
        random_state=random_state,
        init="pca" if n_samples > n_components else "random"
    )
    embedding = tsne.fit_transform(features)
    return embedding


def plot_tsne_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Cluster Projection",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a 2D scatter plot of t-SNE embeddings colored by cluster assignments,
    with noise/anomalies (label -1) highlighted prominently.

    Args:
        embedding: 2D coordinates array of shape (n_samples, 2).
        labels: Cluster or anomaly label assigned to each sample.
        title: Title for the plot.
        save_path: Optional file path to save the generated figure.

    Returns:
        fig: The created Matplotlib figure.
    """
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError(f"Embedding must be of shape (n_samples, 2), got {embedding.shape}")
    if len(labels) != len(embedding):
        raise ValueError(f"Length mismatch: {len(labels)} labels vs {len(embedding)} samples")

    fig, ax = plt.subplots(figsize=(9, 7))

    unique_labels = np.unique(labels)
    # Define a distinct colormap for normal clusters
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_labels))))

    for i, lbl in enumerate(unique_labels):
        mask = (labels == lbl)
        if lbl == -1:
            # Highlight noise/anomalies with bold red 'x' markers
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c="red",
                marker="x",
                s=90,
                linewidths=2,
                label=f"Anomaly (Noise) [n={np.sum(mask)}]",
                zorder=5
            )
        else:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                color=colors[i % len(colors)],
                marker="o",
                alpha=0.8,
                edgecolors="none",
                s=60,
                label=f"Cluster {lbl} [n={np.sum(mask)}]"
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_pca_scree(
    individual_variance: np.ndarray,
    cumulative_variance: np.ndarray,
    threshold: float = 0.95,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generates a dual-axis PCA scree plot showing individual and cumulative explained variance.

    Args:
        individual_variance: Variance ratio explained by each principal component.
        cumulative_variance: Cumulative explained variance ratio.
        threshold: Target variance reference line (e.g., 0.95 = 95%).
        save_path: Optional file path to save the generated figure.

    Returns:
        fig: The created Matplotlib figure.
    """
    n_components = len(individual_variance)
    components = np.arange(1, n_components + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for individual variance
    color_bar = "#4C72B0"
    ax1.bar(components, individual_variance * 100.0, alpha=0.6, color=color_bar, label="Individual Variance (%)")
    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Individual Explained Variance (%)", color=color_bar, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_bar)

    # Line plot on secondary axis for cumulative variance
    ax2 = ax1.twinx()
    color_line = "#C44E52"
    ax2.plot(components, cumulative_variance * 100.0, color=color_line, marker="o", linewidth=2, label="Cumulative Variance (%)")
    ax2.axhline(threshold * 100.0, color="gray", linestyle="--", linewidth=1.5, label=f"Target ({int(threshold*100)}%)")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color=color_line, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim([0, 105])

    plt.title("PCA Explained Variance (Scree Plot)", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_elbow_curve(
    k_values: List[int],
    inertias: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots K-Means inertia against number of clusters (k) for the Elbow method.

    Args:
        k_values: Evaluated k values.
        inertias: Corresponding inertia values.
        save_path: Optional file path to save the generated figure.

    Returns:
        fig: The created Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertias, marker="o", linewidth=2, color="#2B5C8F")
    ax.set_title("K-Means Elbow Method for Optimal $k$", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Number of Clusters ($k$)", fontsize=11)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig




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
