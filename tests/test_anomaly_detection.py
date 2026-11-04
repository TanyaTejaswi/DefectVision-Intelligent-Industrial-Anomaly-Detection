"""
Unit tests for Anomaly Detection module (src/anomaly_detection.py)
"""

import numpy as np
import pytest
from src.anomaly_detection import (
    MemoryBankAnomalyDetector,
    generate_defect_heatmap,
    overlay_heatmap
)


@pytest.fixture
def normal_and_defect_data():
    np.random.seed(42)
    # 60 normal training samples centered at 0
    X_train = np.random.randn(60, 30) * 0.5
    # 20 normal test samples centered at 0
    X_test_normal = np.random.randn(20, 30) * 0.5
    # 10 defect test samples placed far from origin (centered at 10.0)
    X_test_defect = np.random.randn(10, 30) * 0.5 + 10.0

    return X_train, X_test_normal, X_test_defect


def test_memory_bank_anomaly_detector_fit_and_predict(normal_and_defect_data):
    X_train, X_normal, X_defect = normal_and_defect_data

    detector = MemoryBankAnomalyDetector(n_neighbors=2)
    detector.fit(X_train)

    assert detector.memory_bank is not None
    assert detector.memory_bank.shape == X_train.shape
    assert detector.threshold is not None

    # Predict continuous anomaly scores
    scores_normal = detector.predict_anomaly_scores(X_normal)
    scores_defect = detector.predict_anomaly_scores(X_defect)

    assert len(scores_normal) == 20
    assert len(scores_defect) == 10

    # Defect anomaly scores must be strictly higher than normal sample scores
    assert np.mean(scores_defect) > np.mean(scores_normal) * 5.0

    # Binary predictions (0: Normal, 1: Defect)
    preds_normal = detector.predict(X_normal)
    preds_defect = detector.predict(X_defect)

    assert np.sum(preds_defect == 1) == 10  # All 10 defects should be flagged as 1


def test_detector_subsample_ratio(normal_and_defect_data):
    X_train, _, _ = normal_and_defect_data
    detector = MemoryBankAnomalyDetector(subsample_ratio=0.5, random_state=42)
    detector.fit(X_train)

    # 50% of 60 samples = 30 samples in memory bank
    assert detector.memory_bank.shape[0] == 30


def test_generate_defect_heatmap():
    # 14x14 = 196 spatial patch scores
    np.random.seed(42)
    patch_scores = np.random.uniform(0.0, 1.0, size=196)
    # Introduce a localized defect hotspot in the center
    grid = patch_scores.reshape((14, 14))
    grid[6:8, 6:8] += 5.0
    patch_scores = grid.flatten()

    heatmap = generate_defect_heatmap(
        patch_scores,
        grid_size=(14, 14),
        target_size=(224, 224),
        sigma=1.5
    )

    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0
    # The center of the heatmap should have high anomaly intensity
    assert np.mean(heatmap[100:124, 100:124]) > np.mean(heatmap[0:30, 0:30])


def test_overlay_heatmap():
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    dummy_heatmap = np.zeros((100, 100), dtype=float)
    dummy_heatmap[40:60, 40:60] = 1.0

    blended = overlay_heatmap(dummy_image, dummy_heatmap, alpha=0.5)

    assert blended.shape == (100, 100, 3)
    assert blended.dtype == np.uint8
    # Center pixel should have changed due to heatmap overlay
    assert not np.array_equal(blended[50, 50], [128, 128, 128])

