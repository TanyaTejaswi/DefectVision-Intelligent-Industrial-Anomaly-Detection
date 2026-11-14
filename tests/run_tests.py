"""
Comprehensive Test Runner for DefectVision ML & Backend Modules
Executes unit tests across all core components:
- Anomaly Detection (MemoryBankAnomalyDetector, Heatmap Generator)
- Clustering (K-Means, DBSCAN, HDBSCAN)
- Dimensionality Reduction (PCA, Scaling)
- Evaluation (Clustering Metrics, Anomaly Metrics, ROC/PR)
- Visualizations & Overlays
"""

import sys
import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_anomaly_detection():
    print("Testing src.anomaly_detection...", end=" ")
    from src.anomaly_detection import MemoryBankAnomalyDetector, generate_defect_heatmap, overlay_heatmap

    np.random.seed(42)
    X_train = np.random.randn(60, 30) * 0.5
    X_normal = np.random.randn(20, 30) * 0.5
    X_defect = np.random.randn(10, 30) * 0.5 + 10.0

    detector = MemoryBankAnomalyDetector(n_neighbors=2)
    detector.fit(X_train)

    assert detector.memory_bank is not None
    assert detector.memory_bank.shape == X_train.shape
    assert detector.threshold is not None

    scores_normal = detector.predict_anomaly_scores(X_normal)
    scores_defect = detector.predict_anomaly_scores(X_defect)

    assert len(scores_normal) == 20
    assert len(scores_defect) == 10
    assert np.mean(scores_defect) > np.mean(scores_normal) * 5.0

    preds_normal = detector.predict(X_normal)
    preds_defect = detector.predict(X_defect)
    assert np.sum(preds_defect == 1) == 10

    # Subsample ratio test
    det_sub = MemoryBankAnomalyDetector(subsample_ratio=0.5, random_state=42)
    det_sub.fit(X_train)
    assert det_sub.memory_bank.shape[0] == 30

    # Heatmap test
    patch_scores = np.random.uniform(0.0, 1.0, size=196)
    grid = patch_scores.reshape((14, 14))
    grid[6:8, 6:8] += 5.0
    heatmap = generate_defect_heatmap(grid.flatten(), grid_size=(14, 14), target_size=(224, 224), sigma=1.5)
    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0 and np.max(heatmap) <= 1.0

    # Overlay test
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    hm = np.zeros((100, 100), dtype=float)
    hm[40:60, 40:60] = 1.0
    blended = overlay_heatmap(img, hm, alpha=0.5)
    assert blended.shape == (100, 100, 3)
    assert blended.dtype == np.uint8
    print("PASSED")

def test_dimensionality_reduction():
    print("Testing src.dimensionality_reduction...", end=" ")
    from src.dimensionality_reduction import scale_features, apply_pca, compute_explained_variance, find_optimal_components

    np.random.seed(42)
    X = np.random.randn(100, 50)
    scaled, scaler = scale_features(X)
    assert np.allclose(np.mean(scaled, axis=0), 0, atol=1e-5)
    assert np.allclose(np.std(scaled, axis=0), 1, atol=1e-5)

    reduced, pca = apply_pca(scaled, explained_variance_ratio=0.95)
    ind, cum = compute_explained_variance(pca)
    assert cum[-1] >= 0.95
    assert reduced.shape[1] < 50

    k = find_optimal_components(cum, threshold=0.90)
    assert k <= reduced.shape[1]
    print("PASSED")

def test_clustering():
    print("Testing src.clustering...", end=" ")
    from src.clustering import run_kmeans, compute_elbow_inertias, run_dbscan, run_hdbscan, get_cluster_summary

    np.random.seed(42)
    X = np.random.randn(100, 20)
    k_vals, inertias = compute_elbow_inertias(X, k_range=[2, 3, 4])
    assert len(inertias) == 3
    assert inertias[0] > inertias[-1]

    km_labels, _ = run_kmeans(X, n_clusters=4, random_state=42)
    assert len(np.unique(km_labels)) == 4

    db_labels, _ = run_dbscan(X, eps=5.0, min_samples=3)
    sum_db = get_cluster_summary(db_labels)
    assert "num_clusters" in sum_db
    assert "num_anomalies" in sum_db

    hdb_labels, _, _ = run_hdbscan(X, min_cluster_size=5)
    assert len(hdb_labels) == 100
    print("PASSED")

def test_evaluation():
    print("Testing src.evaluation...", end=" ")
    from src.evaluation import compute_clustering_metrics, compute_anomaly_detection_metrics, evaluate_image_level

    np.random.seed(42)
    X = np.random.randn(100, 10)
    labels = np.array([0]*50 + [1]*50)
    cl_metrics = compute_clustering_metrics(X, labels)
    assert "silhouette_score" in cl_metrics
    assert "davies_bouldin_index" in cl_metrics

    y_true = np.array([0]*80 + [1]*20)
    scores = np.array([0.1]*80 + [0.9]*20)
    anom_metrics = compute_anomaly_detection_metrics(y_true, scores)
    assert anom_metrics["roc_auc"] == 1.0
    assert anom_metrics["pr_auc"] == 1.0
    assert anom_metrics["f1_score"] == 1.0

    img_eval = evaluate_image_level(y_true, scores, category_name="test_cat")
    assert img_eval["auroc"] == 1.0
    assert img_eval["true_positives"] == 20
    assert img_eval["true_negatives"] == 80
    print("PASSED")

def test_all_models_exist():
    print("Testing 8-line trained model artifacts...", end=" ")
    from src.anomaly_detection import MemoryBankAnomalyDetector
    categories = ["pcb1", "pcb2", "pcb3", "pcb4", "capsules", "bottle", "cable", "steel"]
    for cat in categories:
        p = PROJECT_ROOT / "models" / f"memory_bank_{cat}.pkl"
        assert p.exists(), f"Missing model: {p}"
        det = MemoryBankAnomalyDetector.load(p)
        assert det.memory_bank is not None
        assert det.memory_bank.ndim == 2
        assert det.optimal_threshold is not None
    print(f"PASSED (All {len(categories)} models loaded and indexed)")

def main():
    print("=" * 65)
    print("DEFECTVISION ML & BACKEND UNIT TEST SUITE")
    print("=" * 65)
    test_anomaly_detection()
    test_dimensionality_reduction()
    test_clustering()
    test_evaluation()
    test_all_models_exist()
    print("=" * 65)
    print("🎉 ALL 5 TEST MODULES PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    main()
