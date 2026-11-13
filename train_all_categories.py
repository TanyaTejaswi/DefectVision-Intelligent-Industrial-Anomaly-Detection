"""
DefectVision — Full Training Pipeline
Trains PatchCore Memory Bank Anomaly Detectors across ALL MVTec AD categories.
Saves: memory bank models, per-category metrics, ROC/PR plots, sample overlays,
       updated anomaly_detection_summary.csv, and a master performance dashboard.

Usage:
    python train_all_categories.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Ensure output directories ───────────────────────────────────────────────
for d in ["models", "reports", "results/plots", "results/metrics"]:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

# ─── PyTorch check ───────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    device_str = "cpu"

from src.preprocessing import build_mvtec_metadata, MVTecDataset, get_default_transforms
from src.anomaly_detection import MemoryBankAnomalyDetector
from src.visualization import plot_anomaly_comparison, overlay_heatmap_on_image
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

# ─── ALL 15 MVTec categories ─────────────────────────────────────────────────
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

SUBSAMPLE_RATIO = 0.1   # Memory-bank coreset fraction (0.1 → 10% of patches kept)
PATCH_GRID      = (28, 28)
IMG_SIZE        = 224


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_image_level(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    category_name: str = "unknown"
) -> Dict:
    """
    Full evaluation metrics for image-level anomaly detection.
    Returns AUROC, AUPRC, optimal-F1 threshold and derived precision/recall/F1,
    plus confusion matrix values.
    """
    auroc = float(roc_auc_score(y_true, y_scores))
    auprc = float(average_precision_score(y_true, y_scores))

    # Find threshold that maximises F1
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0
    )
    best_idx = int(np.argmax(f1s[:-1]))          # last element has no threshold
    optimal_threshold = float(thresholds[best_idx])

    y_pred = (y_scores >= optimal_threshold).astype(int)
    prec  = float(precision_score(y_true, y_pred, zero_division=0))
    rec   = float(recall_score(y_true, y_pred, zero_division=0))
    f1    = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "category":          category_name,
        "auroc":             round(auroc, 4),
        "auprc":             round(auprc, 4),
        "f1_score":          round(f1,   4),
        "precision":         round(prec, 4),
        "recall":            round(rec,  4),
        "optimal_threshold": round(optimal_threshold, 4),
        "true_positives":    int(tp),
        "true_negatives":    int(tn),
        "false_positives":   int(fp),
        "false_negatives":   int(fn),
    }


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    category_name: str = "Category",
    save_path: Optional[Path] = None
):
    """Plot and save combined ROC + PR curve figure."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)

    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curve
    ax1.plot(fpr, tpr, lw=2, color="#E74C3C",
             label=f"ROC-AUC = {auroc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")
    ax1.fill_between(fpr, tpr, alpha=0.08, color="#E74C3C")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title(f"{category_name} — ROC Curve", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.02])

    # PR curve
    ax2.plot(rec, prec, lw=2, color="#2980B9",
             label=f"PR-AUC = {auprc:.4f}")
    ax2.fill_between(rec, prec, alpha=0.08, color="#2980B9")
    baseline = float(np.mean(y_true))
    ax2.axhline(baseline, color="gray", linestyle="--", lw=1, alpha=0.7,
                label=f"Baseline ({baseline:.2f})")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title(f"{category_name} — Precision-Recall Curve",
                  fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    ✔ ROC/PR curves saved → {save_path.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PER-CATEGORY TRAINING + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_category(
    category: str,
    metadata_df: pd.DataFrame,
    extractor,
    device: str = "cpu",
) -> Dict:
    """Train memory bank and evaluate for a single MVTec category."""

    print(f"\n{'='*65}")
    print(f"  🏭  Category: {category.upper()}")
    print(f"{'='*65}")

    transforms = get_default_transforms(img_size=IMG_SIZE, crop_size=IMG_SIZE)

    train_ds = MVTecDataset(metadata_df, transform=transforms,
                            target_split="train", target_category=category)
    test_ds  = MVTecDataset(metadata_df, transform=transforms,
                            target_split="test",  target_category=category)

    n_train = len(train_ds)
    n_test  = len(test_ds)
    print(f"  Train (normal): {n_train} images | Test: {n_test} images")

    if n_train == 0:
        print(f"  ⚠ No training images found for {category}. Skipping.")
        return {}

    # ── 1. Extract patch features from training set ──────────────────────────
    print(f"  Extracting patch features from {n_train} training images …")
    t0 = time.time()
    train_patches = []

    for idx in range(n_train):
        item = train_ds[idx]
        img  = item["image"]
        patches = extractor.extract_patch_features(img, target_size=PATCH_GRID)
        p_np = (patches.squeeze(0)
                       .permute(1, 2, 0)
                       .reshape(-1, patches.shape[1])
                       .cpu().numpy())
        train_patches.append(p_np)

        if (idx + 1) % 50 == 0 or (idx + 1) == n_train:
            elapsed = time.time() - t0
            print(f"    … {idx+1}/{n_train} images "
                  f"({elapsed:.1f}s elapsed)")

    all_patches = np.vstack(train_patches)
    print(f"  Total patches: {all_patches.shape[0]:,} × {all_patches.shape[1]} dims  "
          f"[{time.time()-t0:.1f}s]")

    # ── 2. Build & fit memory bank ───────────────────────────────────────────
    detector = MemoryBankAnomalyDetector(
        category=category,
        n_neighbors=1,
        subsample_ratio=SUBSAMPLE_RATIO,
        gaussian_sigma=4.0,
        random_state=42
    )
    detector.fit(all_patches)
    print(f"  Memory bank: {detector.memory_bank.shape[0]:,} vectors retained "
          f"({SUBSAMPLE_RATIO*100:.0f}% coreset)")

    # ── 3. Score test set ────────────────────────────────────────────────────
    print(f"  Scoring {n_test} test images …")
    y_true, y_scores = [], []
    sample_pool      = []   # for visual comparison plots

    for idx in range(n_test):
        item        = test_ds[idx]
        img         = item["image"]
        is_anomaly  = item["is_anomaly"]
        defect_type = item["defect_type"]

        patches = extractor.extract_patch_features(img, target_size=PATCH_GRID)
        p_np    = (patches.squeeze(0)
                          .permute(1, 2, 0)
                          .reshape(-1, patches.shape[1])
                          .cpu().numpy())

        score, heatmap = detector.predict_image_and_map(
            p_np, spatial_grid=PATCH_GRID, target_size=(IMG_SIZE, IMG_SIZE)
        )
        y_true.append(is_anomaly)
        y_scores.append(score)

        # Collect up to 1 normal + 2 defect samples for plots
        n_normal  = sum(1 for s in sample_pool if s["is_anomaly"] == 0)
        n_defect  = sum(1 for s in sample_pool if s["is_anomaly"] == 1)
        if (is_anomaly == 0 and n_normal < 1) or (is_anomaly == 1 and n_defect < 2):
            raw_img  = Image.open(item["image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            mask_obj = item["mask"]
            if TORCH_AVAILABLE and hasattr(mask_obj, "cpu"):
                mask_np = mask_obj.squeeze(0).cpu().numpy()
            else:
                mask_np = np.array(mask_obj) if mask_obj is not None else np.zeros((IMG_SIZE, IMG_SIZE))
            sample_pool.append({
                "image": raw_img, "mask": mask_np,
                "heatmap": heatmap, "score": score,
                "defect_type": defect_type, "is_anomaly": is_anomaly
            })

    y_true   = np.array(y_true)
    y_scores = np.array(y_scores)

    # ── 4. Compute metrics ───────────────────────────────────────────────────
    metrics = evaluate_image_level(y_true, y_scores, category_name=category)
    detector.optimal_threshold = metrics["optimal_threshold"]

    print(f"\n  ── Benchmark Results ──")
    print(f"     AUROC  : {metrics['auroc']*100:.2f}%")
    print(f"     AUPRC  : {metrics['auprc']*100:.2f}%")
    print(f"     F1     : {metrics['f1_score']:.4f}")
    print(f"     Prec   : {metrics['precision']*100:.2f}%")
    print(f"     Recall : {metrics['recall']*100:.2f}%")
    print(f"     Thresh : {metrics['optimal_threshold']:.4f}")
    print(f"     CM     : TP={metrics['true_positives']} | TN={metrics['true_negatives']} "
          f"| FP={metrics['false_positives']} | FN={metrics['false_negatives']}")

    # ── 5. Save memory bank model ────────────────────────────────────────────
    model_path = PROJECT_ROOT / "models" / f"memory_bank_{category}.pkl"
    detector.save(model_path)

    # ── 6. Save JSON metrics ─────────────────────────────────────────────────
    json_path = PROJECT_ROOT / "reports" / f"{category}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 7. ROC & PR curves ───────────────────────────────────────────────────
    plot_roc_pr_curves(
        y_true, y_scores,
        category_name=category.title(),
        save_path=PROJECT_ROOT / "results" / "plots" / f"{category}_roc_pr.png"
    )

    # ── 8. Sample inspection overlays ───────────────────────────────────────
    for i, comp in enumerate(sample_pool):
        plot_anomaly_comparison(
            comp["image"], comp["mask"], comp["heatmap"], comp["score"],
            defect_type=comp["defect_type"], category=category,
            save_path=PROJECT_ROOT / "results" / "plots" / f"{category}_sample_{i}.png"
        )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD: Updated summary charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_master_dashboard(summary_df: pd.DataFrame, save_dir: Path):
    """
    Generate a comprehensive performance dashboard:
      - AUROC bar chart across all categories
      - AUPRC bar chart
      - F1-Score bar chart
      - Precision vs Recall scatter
      - Confusion matrix heatmap (TP/TN/FP/FN totals)
    """
    cats = summary_df["category"].tolist()
    n    = len(cats)
    idx  = np.arange(n)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.95, n))

    # ── 1. AUROC ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 5))
    bars = ax.bar(idx, summary_df["auroc"] * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(idx); ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=10)
    ax.set_ylim([0, 105])
    ax.set_ylabel("AUROC (%)", fontsize=12)
    ax.set_title("AUROC — All MVTec Categories", fontsize=14, fontweight="bold")
    ax.axhline(100, color="gray", linestyle="--", lw=1, alpha=0.5, label="Perfect (100%)")
    ax.axhline(summary_df["auroc"].mean() * 100, color="#E74C3C",
               linestyle="-.", lw=1.5, label=f"Mean ({summary_df['auroc'].mean()*100:.1f}%)")
    for bar, val in zip(bars, summary_df["auroc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "dashboard_auroc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ dashboard_auroc.png")

    # ── 2. AUPRC ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 5))
    bars = ax.bar(idx, summary_df["auprc"] * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(idx); ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=10)
    ax.set_ylim([0, 105])
    ax.set_ylabel("AUPRC (%)", fontsize=12)
    ax.set_title("AUPRC — All MVTec Categories", fontsize=14, fontweight="bold")
    ax.axhline(summary_df["auprc"].mean() * 100, color="#2980B9",
               linestyle="-.", lw=1.5, label=f"Mean ({summary_df['auprc'].mean()*100:.1f}%)")
    for bar, val in zip(bars, summary_df["auprc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "dashboard_auprc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ dashboard_auprc.png")

    # ── 3. F1 Score ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 5))
    bars = ax.bar(idx, summary_df["f1_score"] * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(idx); ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=10)
    ax.set_ylim([0, 110])
    ax.set_ylabel("F1-Score (%)", fontsize=12)
    ax.set_title("F1-Score — All MVTec Categories", fontsize=14, fontweight="bold")
    ax.axhline(summary_df["f1_score"].mean() * 100, color="#8E44AD",
               linestyle="-.", lw=1.5, label=f"Mean ({summary_df['f1_score'].mean()*100:.1f}%)")
    for bar, val in zip(bars, summary_df["f1_score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "dashboard_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ dashboard_f1.png")

    # ── 4. Precision vs Recall scatter ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        summary_df["recall"] * 100,
        summary_df["precision"] * 100,
        c=summary_df["f1_score"],
        cmap="RdYlGn", s=160, edgecolors="k", linewidth=0.5, zorder=3
    )
    for _, row in summary_df.iterrows():
        ax.annotate(row["category"], (row["recall"] * 100, row["precision"] * 100),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("F1-Score", fontsize=10)
    ax.set_xlabel("Recall (%)", fontsize=12)
    ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("Precision vs Recall (colour = F1-Score)", fontsize=13, fontweight="bold")
    ax.set_xlim([0, 110]); ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "dashboard_prec_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ dashboard_prec_recall.png")

    # ── 5. Confusion matrix totals heatmap ───────────────────────────────────
    cm_data = summary_df[["true_positives", "false_negatives",
                           "false_positives", "true_negatives"]].values
    cm_matrix = cm_data.reshape(-1, 2, 2)  # (n_cats, 2, 2)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    for i, (ax, cat) in enumerate(zip(axes, cats)):
        cm = cm_matrix[i]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Defect"], fontsize=8)
        ax.set_yticklabels(["True Normal", "True Defect"], fontsize=8)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]),
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[r, c] > cm.max() * 0.6 else "black")
        ax.set_title(cat.title(), fontsize=10, fontweight="bold")
    for ax in axes[n:]:
        ax.axis("off")
    plt.suptitle("Confusion Matrices — All Categories", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_dir / "dashboard_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ dashboard_confusion_matrices.png")

    # ── 6. Combined metrics comparison (line chart) ──────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, n * 0.95), 5))
    ax.plot(idx, summary_df["auroc"]    * 100, "o-", lw=2, color="#E74C3C", label="AUROC")
    ax.plot(idx, summary_df["auprc"]    * 100, "s-", lw=2, color="#2980B9", label="AUPRC")
    ax.plot(idx, summary_df["f1_score"] * 100, "^-", lw=2, color="#27AE60", label="F1-Score")
    ax.plot(idx, summary_df["precision"]* 100, "D-", lw=1.5, color="#8E44AD", alpha=0.7, label="Precision")
    ax.plot(idx, summary_df["recall"]   * 100, "P-", lw=1.5, color="#F39C12", alpha=0.7, label="Recall")
    ax.set_xticks(idx); ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=10)
    ax.set_ylim([0, 110])
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Full Performance Comparison — All MVTec Categories",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=5); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "dashboard_all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ dashboard_all_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  DEFECTVISION — FULL TRAINING PIPELINE (ALL 15 CATEGORIES)")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("❌  PyTorch is not available. Cannot run deep feature extraction.")
        sys.exit(1)

    print(f"  Device: {device_str.upper()}")
    print(f"  Coreset ratio: {SUBSAMPLE_RATIO*100:.0f}%")

    # Load feature extractor (ResNet50 pretrained, frozen)
    from src.feature_extraction import DeepFeatureExtractor
    extractor = DeepFeatureExtractor(backbone_name="resnet50", device=device_str)

    # Build / load metadata for all categories
    dataset_root = PROJECT_ROOT / "data" / "mvtec"
    if not dataset_root.exists():
        print(f"❌  Dataset not found at: {dataset_root}")
        sys.exit(1)

    metadata_path = PROJECT_ROOT / "data" / "embeddings" / "metadata.csv"
    if metadata_path.exists():
        print(f"\n  Loading existing metadata from {metadata_path.name} …")
        metadata_df = pd.read_csv(metadata_path)
    else:
        print("\n  Building metadata index from data/mvtec …")
        metadata_df = build_mvtec_metadata(dataset_root, categories=ALL_CATEGORIES)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(metadata_path, index=False)

    # Fix: CSV reads empty mask_path cells as float NaN → convert to None/str
    metadata_df["mask_path"] = metadata_df["mask_path"].where(
        metadata_df["mask_path"].notna(), other=None
    )

    total_imgs = len(metadata_df)
    print(f"  Total images in metadata: {total_imgs:,} across {len(ALL_CATEGORIES)} categories\n")

    # ── Train all categories ─────────────────────────────────────────────────
    all_metrics = []
    failed      = []
    t_global    = time.time()

    for cat in ALL_CATEGORIES:
        try:
            metrics = run_category(
                category=cat,
                metadata_df=metadata_df,
                extractor=extractor,
                device=device_str,
            )
            if metrics:
                all_metrics.append(metrics)
        except Exception as exc:
            print(f"\n  ❌  FAILED for {cat}: {exc}")
            failed.append(cat)

    total_time = time.time() - t_global
    print(f"\n  Total training time: {total_time/60:.1f} minutes")

    if not all_metrics:
        print("❌  No categories were successfully trained.")
        sys.exit(1)

    # ── Compile & save summary CSV ───────────────────────────────────────────
    summary_df = pd.DataFrame(all_metrics)

    # Reorder columns to match original summary format
    col_order = ["category", "auroc", "auprc", "f1_score", "precision", "recall",
                 "optimal_threshold", "true_positives", "true_negatives",
                 "false_positives", "false_negatives"]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    # Also write the slim version that matches the existing reports/anomaly_detection_summary.csv
    slim_df = summary_df[["category", "auroc", "auprc", "f1_score",
                           "precision", "recall", "optimal_threshold"]].copy()
    slim_df.columns = ["category", "auroc", "auprc", "f1_score",
                       "precision", "recall", "optimal_threshold"]
    slim_path = PROJECT_ROOT / "reports" / "anomaly_detection_summary.csv"
    slim_df.to_csv(slim_path, index=False)
    print(f"\n  ✅  Updated summary → {slim_path}")

    full_path = PROJECT_ROOT / "reports" / "anomaly_detection_summary_full.csv"
    summary_df.to_csv(full_path, index=False)
    print(f"  ✅  Full report → {full_path}")

    # ── Build performance dashboard ──────────────────────────────────────────
    print("\n  Generating master performance dashboard …")
    plot_master_dashboard(summary_df, save_dir=PROJECT_ROOT / "results" / "plots")

    # ── Print scoreboard ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🏆  MULTI-CATEGORY ANOMALY DETECTION SCOREBOARD")
    print("=" * 70)
    display_cols = ["category", "auroc", "auprc", "f1_score", "precision", "recall"]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 70)

    mean_auroc = summary_df["auroc"].mean()
    mean_auprc = summary_df["auprc"].mean()
    mean_f1    = summary_df["f1_score"].mean()
    print(f"\n  Mean AUROC : {mean_auroc*100:.2f}%")
    print(f"  Mean AUPRC : {mean_auprc*100:.2f}%")
    print(f"  Mean F1    : {mean_f1*100:.2f}%")

    if failed:
        print(f"\n  ⚠  Failed categories: {', '.join(failed)}")

    print(f"\n  ✅  Models saved in : {PROJECT_ROOT / 'models'}")
    print(f"  ✅  Reports saved in: {PROJECT_ROOT / 'reports'}")
    print(f"  ✅  Plots saved in  : {PROJECT_ROOT / 'results' / 'plots'}")
    print("\n  Pipeline complete!\n")


if __name__ == "__main__":
    main()
