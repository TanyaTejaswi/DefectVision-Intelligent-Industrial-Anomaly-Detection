"""
Unit tests for Dimensionality Reduction module (src/dimensionality_reduction.py)
"""

import numpy as np
import pytest
from src.dimensionality_reduction import (
    scale_features,
    apply_pca,
    compute_explained_variance,
    find_optimal_components
)


@pytest.fixture
def sample_feature_matrix():
    """Generates a reproducible synthetic feature matrix simulating deep features."""
    np.random.seed(42)
    # 50 samples with 100 features
    return np.random.randn(50, 100)


def test_scale_features(sample_feature_matrix):
    scaled, scaler = scale_features(sample_feature_matrix)
    assert scaled.shape == sample_feature_matrix.shape
    # Mean of scaled features should be approximately 0
    np.testing.assert_allclose(np.mean(scaled, axis=0), 0.0, atol=1e-7)
    # Variance of scaled features should be approximately 1
    np.testing.assert_allclose(np.var(scaled, axis=0), 1.0, atol=1e-7)


def test_apply_pca_with_fixed_components(sample_feature_matrix):
    n_components = 15
    reduced, pca = apply_pca(sample_feature_matrix, n_components=n_components)
    assert reduced.shape == (50, n_components)
    assert hasattr(pca, "components_")
    assert pca.n_components_ == n_components


def test_apply_pca_with_variance_ratio(sample_feature_matrix):
    target_variance = 0.85
    reduced, pca = apply_pca(sample_feature_matrix, explained_variance_ratio=target_variance)
    assert reduced.shape[0] == 50
    assert reduced.shape[1] <= 50  # Must be <= min(n_samples, n_features)
    assert np.sum(pca.explained_variance_ratio_) >= target_variance


def test_compute_explained_variance(sample_feature_matrix):
    reduced, pca = apply_pca(sample_feature_matrix, n_components=10)
    ind_var, cum_var = compute_explained_variance(pca)
    assert len(ind_var) == 10
    assert len(cum_var) == 10
    # Cumulative variance should be monotonically increasing
    assert np.all(np.diff(cum_var) >= 0)
    # Final cumulative value should match the sum of individual ratios
    np.testing.assert_allclose(cum_var[-1], np.sum(ind_var))


def test_find_optimal_components():
    cum_var = np.array([0.40, 0.65, 0.82, 0.91, 0.96, 0.99])
    # For 90% threshold, component index 3 (0.91) is 4th component (1-indexed)
    opt = find_optimal_components(cum_var, threshold=0.90)
    assert opt == 4
    # For 95% threshold, 5th component (0.96)
    opt_95 = find_optimal_components(cum_var, threshold=0.95)
    assert opt_95 == 5

