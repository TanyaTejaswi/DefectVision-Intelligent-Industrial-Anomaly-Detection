"""
Evaluation Module
Responsibility: Gayatri U (Clustering Metrics) & Tanya (Anomaly Metrics)

This module is strictly organized into two distinct evaluation categories:
1. Clustering Evaluation Metrics (Unsupervised / Intrinsic)
   - Evaluates the geometrical cohesion and separation of clusters without labels.
2. Anomaly Detection Metrics (Supervised Validation / Extrinsic)
   - Evaluates detection accuracy against ground-truth normal vs. defective labels.
3. Post-Hoc Alignment Metrics (Optional)
   - ARI / NMI for post-hoc analysis when ground-truth categories are known.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    adjusted_rand_score,
    normalized_mutual_info_score
)


# ============================================================================
# 1. CLUSTERING EVALUATION METRICS (UNSUPERVISED / INTRINSIC)
# ============================================================================

def compute_clustering_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    exclude_noise: bool = True
) -> Dict[str, float]:
    """
    Computes intrinsic clustering quality metrics for unsupervised cluster validation.

    Metrics calculated:
        - Silhouette Score: Measures cluster cohesion vs separation [-1 to 1] (higher is better).
        - Davies-Bouldin Index: Ratio of within-cluster scatter to separation (lower is better, >= 0).
        - Calinski-Harabasz Index: Variance ratio criterion (higher is better).

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Cluster assignment array of shape (n_samples,).
        exclude_noise: If True, samples labeled -1 (DBSCAN/HDBSCAN noise) are excluded
                       from the geometrical cluster quality computation.

    Returns:
        metrics: Dictionary containing 'silhouette_score', 'davies_bouldin_index',
                 'calinski_harabasz_index', and 'evaluated_samples'.
    """
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    if len(features) != len(labels):
        raise ValueError(f"Length mismatch: {len(features)} samples vs {len(labels)} labels")

    # Filter noise points if requested
    if exclude_noise:
        valid_mask = (labels != -1)
        valid_features = features[valid_mask]
        valid_labels = labels[valid_mask]
    else:
        valid_features = features
        valid_labels = labels

    n_valid = len(valid_labels)
    unique_labels = np.unique(valid_labels)
    n_clusters = len(unique_labels)

    # Intrinsic metrics require at least 2 clusters and strictly fewer clusters than samples
    if n_clusters < 2 or n_clusters >= n_valid:
        return {
            "silhouette_score": float("nan"),
            "davies_bouldin_index": float("nan"),
            "calinski_harabasz_index": float("nan"),
            "evaluated_samples": n_valid,
            "evaluated_clusters": n_clusters
        }

    sil = float(silhouette_score(valid_features, valid_labels))
    db = float(davies_bouldin_score(valid_features, valid_labels))
    ch = float(calinski_harabasz_score(valid_features, valid_labels))

    return {
        "silhouette_score": sil,
        "davies_bouldin_index": db,
        "calinski_harabasz_index": ch,
        "evaluated_samples": n_valid,
        "evaluated_clusters": n_clusters
    }


# ============================================================================
# 2. ANOMALY DETECTION METRICS (SUPERVISED VALIDATION / EXTRINSIC)
# ============================================================================

def compute_anomaly_detection_metrics(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluates anomaly detection performance against ground-truth labels (0: Normal, 1: Defect).

    Metrics calculated:
        - ROC-AUC: Area under the Receiver Operating Characteristic curve.
        - PR-AUC: Area under the Precision-Recall curve (ideal for imbalanced industrial defects).
        - Precision, Recall, F1-Score: Evaluated at the specified threshold (or optimal F1 cutoff).

    Args:
        y_true: Binary ground-truth labels (0 for normal, 1 for defective).
        anomaly_scores: Continuous anomaly scores or distance metrics for each sample.
        threshold: Optional cutoff threshold to binarize scores. If None, picks threshold
                   maximizing the F1-score across candidate thresholds.

    Returns:
        metrics: Dictionary containing 'roc_auc', 'pr_auc', 'threshold',
                 'precision', 'recall', and 'f1_score'.
    """
    y_true = np.asarray(y_true).astype(int)
    anomaly_scores = np.asarray(anomaly_scores).astype(float)

    if len(y_true) != len(anomaly_scores):
        raise ValueError("Length mismatch between y_true and anomaly_scores")

    # Global ranking metrics
    roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    pr_auc = float(average_precision_score(y_true, anomaly_scores))

    # Determine threshold
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, anomaly_scores)
        f1_scores = np.zeros_like(thresholds)
        for i in range(len(thresholds)):
            p = precisions[i]
            r = recalls[i]
            f1_scores[i] = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        if len(thresholds) > 0 and np.max(f1_scores) > 0:
            best_idx = int(np.argmax(f1_scores))
            chosen_threshold = float(thresholds[best_idx])
        else:
            chosen_threshold = float(np.median(anomaly_scores))
    else:
        chosen_threshold = float(threshold)

    y_pred = (anomaly_scores >= chosen_threshold).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": chosen_threshold,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }


# ============================================================================
# 3. POST-HOC CLUSTER ALIGNMENT METRICS
# ============================================================================

def compute_posthoc_cluster_alignment(
    labels_true: np.ndarray,
    labels_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes external cluster agreement metrics when category labels are available post-hoc.
    (Note: Used only for retrospective analysis, never for training).

    Args:
        labels_true: Ground-truth category/class labels.
        labels_pred: Discovered cluster labels.

    Returns:
        metrics: Dictionary containing 'adjusted_rand_index' (ARI) and
                 'normalized_mutual_info' (NMI).
    """
    ari = float(adjusted_rand_score(labels_true, labels_pred))
    nmi = float(normalized_mutual_info_score(labels_true, labels_pred))

    return {
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi
    }


def evaluate_image_level(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    category_name: str = "general",
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Standard image-level anomaly detection evaluation wrapper.
    Computes AUROC, AUPRC, F1-score, Confusion Matrix stats, and optimal threshold.
    """
    m = compute_anomaly_detection_metrics(y_true, anomaly_scores, threshold=threshold)
    y_true_bin = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(anomaly_scores) >= m["threshold"]).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred == 0)))

    return {
        "category": category_name,
        "auroc": m["roc_auc"],
        "auprc": m["pr_auc"],
        "f1_score": m["f1_score"],
        "precision": m["precision"],
        "recall": m["recall"],
        "optimal_threshold": m["threshold"],
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }


def plot_roc_pr_curves(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    category_name: str = "Industrial Component",
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plots high-resolution dual ROC and Precision-Recall evaluation curves.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    prec, rec, _ = precision_recall_curve(y_true, anomaly_scores)
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#059669", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=10, fontweight="bold")
    axes[0].set_title(f"{category_name} - Receiver Operating Characteristic", fontsize=11, fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # PR Curve
    axes[1].plot(rec, prec, color="#2563eb", lw=2.5, label=f"PR (AUC = {pr_auc:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Precision", fontsize=10, fontweight="bold")
    axes[1].set_title(f"{category_name} - Precision-Recall Curve", fontsize=11, fontweight="bold")
    axes[1].legend(loc="lower left", frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig

