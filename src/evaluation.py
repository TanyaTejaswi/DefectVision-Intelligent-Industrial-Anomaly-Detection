"""
Evaluation Module
Computes performance metrics for visual anomaly detection:
- Image-level AUROC and AUPRC (Average Precision)
- Pixel-level AUROC (Localization accuracy against ground truth masks)
- Optimal decision thresholds, Confusion Matrix, Precision, Recall, and F1-score
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_image_level(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    category_name: str = "all"
) -> Dict[str, float]:
    """
    Computes image-level AUROC, AUPRC, and optimal threshold metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    # 1. ROC and AUROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)

    # 2. Precision-Recall and AUPRC
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(recalls, precisions)

    # 3. Find optimal threshold using Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(roc_thresholds[best_idx])

    # 4. Binary predictions at optimal threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results = {
        "category": category_name,
        "auroc": float(auroc),
        "auprc": float(auprc),
        "optimal_threshold": best_threshold,
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    }

    return results


def evaluate_pixel_level(
    ground_truth_masks: List[np.ndarray],
    anomaly_maps: List[np.ndarray],
    category_name: str = "all"
) -> Dict[str, float]:
    """
    Computes pixel-level AUROC by flattening all ground truth masks and predicted maps.
    """
    flat_gt = []
    flat_maps = []

    for gt, pred in zip(ground_truth_masks, anomaly_maps):
        flat_gt.append((gt.flatten() > 0.5).astype(int))
        flat_maps.append(pred.flatten().astype(float))

    all_gt = np.concatenate(flat_gt)
    all_pred = np.concatenate(flat_maps)

    pixel_auroc = roc_auc_score(all_gt, all_pred)
    precisions, recalls, _ = precision_recall_curve(all_gt, all_pred)
    pixel_auprc = auc(recalls, precisions)

    return {
        "category": category_name,
        "pixel_auroc": float(pixel_auroc),
        "pixel_auprc": float(pixel_auprc)
    }


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    category_name: str = "Category",
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plots side-by-side ROC Curve and Precision-Recall Curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUROC = {auroc:.4f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax1.set_title(f"ROC Curve - {category_name}", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Precision-Recall Curve
    ax2.plot(recall, precision, color="forestgreen", lw=2, label=f"PR curve (AUPRC = {auprc:.4f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title(f"Precision-Recall Curve - {category_name}", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved evaluation curves to {save_path}")

    return fig
