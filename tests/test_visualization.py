"""
Unit tests for Visualization module (src/visualization.py)
"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from src.visualization import (
    compute_tsne_embedding,
    plot_tsne_clusters,
    plot_pca_scree,
    plot_elbow_curve
)


@pytest.fixture
def sample_embedding_and_labels():
    np.random.seed(42)
    embedding = np.random.randn(40, 2)
    # 3 normal clusters (0, 1, 2) and 2 anomalies (-1)
    labels = np.array([0] * 15 + [1] * 15 + [2] * 8 + [-1] * 2)
    return embedding, labels


def test_compute_tsne_embedding():
    np.random.seed(42)
    features = np.random.randn(30, 20)
    embedding = compute_tsne_embedding(features, n_components=2, perplexity=10.0, max_iter=250)

    assert embedding.shape == (30, 2)
    assert not np.isnan(embedding).any()


def test_plot_tsne_clusters(sample_embedding_and_labels, tmp_path):
    embedding, labels = sample_embedding_and_labels
    save_file = tmp_path / "test_tsne.png"

    fig = plot_tsne_clusters(embedding, labels, title="Test Projection", save_path=str(save_file))
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)
    assert os.path.getsize(save_file) > 0
    plt.close(fig)


def test_plot_pca_scree(tmp_path):
    ind_var = np.array([0.45, 0.25, 0.15, 0.08, 0.07])
    cum_var = np.cumsum(ind_var)
    save_file = tmp_path / "test_scree.png"

    fig = plot_pca_scree(ind_var, cum_var, threshold=0.90, save_path=str(save_file))
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)
    assert os.path.getsize(save_file) > 0
    plt.close(fig)


def test_plot_elbow_curve(tmp_path):
    k_vals = [2, 3, 4, 5, 6]
    inertias = [1200.0, 750.0, 480.0, 390.0, 350.0]
    save_file = tmp_path / "test_elbow.png"

    fig = plot_elbow_curve(k_vals, inertias, save_path=str(save_file))
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)
    assert os.path.getsize(save_file) > 0
    plt.close(fig)

