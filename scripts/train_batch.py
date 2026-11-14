"""
DefectVision — Batch Training Script
Trains a specified subset of categories.
Usage: python train_batch.py cable capsule carpet grid hazelnut
"""

import json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for d in ["models", "reports", "results/plots", "results/metrics"]:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

try:
    import torch
    TORCH_AVAILABLE = True
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    device_str = "cpu"

from src.preprocessing import build_mvtec_metadata, MVTecDataset, get_default_transforms
from src.anomaly_detection import MemoryBankAnomalyDetector
from src.visualization import plot_anomaly_comparison
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

SUBSAMPLE_RATIO = 0.1
PATCH_GRID = (28, 28)
IMG_SIZE   = 224


def evaluate_image_level(y_true, y_scores, category_name="unknown"):
    auroc = float(roc_auc_score(y_true, y_scores))
    auprc = float(average_precision_score(y_true, y_scores))
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls), 0.0
    )
    best_idx = int(np.argmax(f1s[:-1]))
    optimal_threshold = float(thresholds[best_idx])
    y_pred = (y_scores >= optimal_threshold).astype(int)
    prec  = float(precision_score(y_true, y_pred, zero_division=0))
    rec   = float(recall_score(y_true, y_pred, zero_division=0))
    f1    = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "category": category_name,
        "auroc": round(auroc, 4), "auprc": round(auprc, 4),
        "f1_score": round(f1, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "optimal_threshold": round(optimal_threshold, 4),
        "true_positives": int(tp), "true_negatives": int(tn),
        "false_positives": int(fp), "false_negatives": int(fn),
    }


def plot_roc_pr_curves(y_true, y_scores, category_name="Category", save_path=None):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(fpr, tpr, lw=2, color="#E74C3C", label=f"ROC-AUC = {auroc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.fill_between(fpr, tpr, alpha=0.08, color="#E74C3C")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title(f"{category_name} — ROC Curve", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10); ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.02])

    ax2.plot(rec, prec, lw=2, color="#2980B9", label=f"PR-AUC = {auprc:.4f}")
    ax2.fill_between(rec, prec, alpha=0.08, color="#2980B9")
    ax2.axhline(float(np.mean(y_true)), color="gray", linestyle="--", lw=1)
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title(f"{category_name} — Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10); ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    ✔ ROC/PR → {Path(save_path).name}")
    plt.close(fig)


def run_category(category, metadata_df, extractor):
    print(f"\n{'='*65}")
    print(f"  🏭  Category: {category.upper()}")
    print(f"{'='*65}")
    sys.stdout.flush()

    transforms = get_default_transforms(img_size=IMG_SIZE, crop_size=IMG_SIZE)
    train_ds = MVTecDataset(metadata_df, transform=transforms, target_split="train",  target_category=category)
    test_ds  = MVTecDataset(metadata_df, transform=transforms, target_split="test",   target_category=category)

    n_train, n_test = len(train_ds), len(test_ds)
    print(f"  Train (normal): {n_train} | Test: {n_test}")
    sys.stdout.flush()

    if n_train == 0:
        print(f"  ⚠ No training images. Skipping.")
        return {}

    # 1. Extract patch features
    print(f"  Extracting patch features …")
    sys.stdout.flush()
    t0 = time.time()
    train_patches = []

    for idx in range(n_train):
        item = train_ds[idx]
        patches = extractor.extract_patch_features(item["image"], target_size=PATCH_GRID)
        p_np = patches.squeeze(0).permute(1, 2, 0).reshape(-1, patches.shape[1]).cpu().numpy()
        train_patches.append(p_np)
        if (idx + 1) % 50 == 0 or (idx + 1) == n_train:
            print(f"    … {idx+1}/{n_train}  ({time.time()-t0:.1f}s)")
            sys.stdout.flush()

    all_patches = np.vstack(train_patches)
    print(f"  Patches collected: {all_patches.shape[0]:,} × {all_patches.shape[1]}  [{time.time()-t0:.1f}s]")
    sys.stdout.flush()

    # 2. Memory bank
    detector = MemoryBankAnomalyDetector(
        category=category, n_neighbors=1,
        subsample_ratio=SUBSAMPLE_RATIO, gaussian_sigma=4.0, random_state=42
    )
    detector.fit(all_patches)
    print(f"  Memory bank vectors: {detector.memory_bank.shape[0]:,}")
    sys.stdout.flush()

    # 3. Test scoring
    print(f"  Scoring {n_test} test images …")
    sys.stdout.flush()
    y_true, y_scores, sample_pool = [], [], []

    for idx in range(n_test):
        item = test_ds[idx]
        patches = extractor.extract_patch_features(item["image"], target_size=PATCH_GRID)
        p_np = patches.squeeze(0).permute(1, 2, 0).reshape(-1, patches.shape[1]).cpu().numpy()
        score, heatmap = detector.predict_image_and_map(p_np, spatial_grid=PATCH_GRID, target_size=(IMG_SIZE, IMG_SIZE))
        y_true.append(item["is_anomaly"]); y_scores.append(score)

        n_norm = sum(1 for s in sample_pool if s["is_anomaly"]==0)
        n_def  = sum(1 for s in sample_pool if s["is_anomaly"]==1)
        if (item["is_anomaly"]==0 and n_norm<1) or (item["is_anomaly"]==1 and n_def<2):
            raw_img = Image.open(item["image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            mask_obj = item["mask"]
            if TORCH_AVAILABLE and hasattr(mask_obj, "cpu"):
                mask_np = mask_obj.squeeze(0).cpu().numpy()
            else:
                mask_np = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            sample_pool.append({"image": raw_img, "mask": mask_np, "heatmap": heatmap,
                                 "score": score, "defect_type": item["defect_type"],
                                 "is_anomaly": item["is_anomaly"]})

    y_true, y_scores = np.array(y_true), np.array(y_scores)

    # 4. Metrics
    metrics = evaluate_image_level(y_true, y_scores, category_name=category)
    detector.optimal_threshold = metrics["optimal_threshold"]

    print(f"\n  Results for {category.upper()}:")
    print(f"    AUROC  : {metrics['auroc']*100:.2f}%")
    print(f"    AUPRC  : {metrics['auprc']*100:.2f}%")
    print(f"    F1     : {metrics['f1_score']*100:.2f}%")
    print(f"    Prec   : {metrics['precision']*100:.2f}%")
    print(f"    Recall : {metrics['recall']*100:.2f}%")
    print(f"    Thresh : {metrics['optimal_threshold']:.4f}")
    print(f"    CM     : TP={metrics['true_positives']} TN={metrics['true_negatives']} FP={metrics['false_positives']} FN={metrics['false_negatives']}")
    sys.stdout.flush()

    # 5. Save model
    model_path = PROJECT_ROOT / "models" / f"memory_bank_{category}.pkl"
    detector.save(model_path)

    # 6. JSON metrics
    with open(PROJECT_ROOT / "reports" / f"{category}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 7. ROC/PR curves
    plot_roc_pr_curves(y_true, y_scores, category_name=category.title(),
                       save_path=PROJECT_ROOT / "results" / "plots" / f"{category}_roc_pr.png")

    # 8. Sample overlays
    for i, comp in enumerate(sample_pool):
        plot_anomaly_comparison(
            comp["image"], comp["mask"], comp["heatmap"], comp["score"],
            defect_type=comp["defect_type"], category=category,
            save_path=PROJECT_ROOT / "results" / "plots" / f"{category}_sample_{i}.png"
        )
    return metrics


def rebuild_summary():
    """Read all per-category JSON metrics and rebuild the master CSV."""
    rows = []
    for f in sorted((PROJECT_ROOT / "reports").glob("*_metrics.json")):
        with open(f) as fp:
            rows.append(json.load(fp))
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = ["category", "auroc", "auprc", "f1_score", "precision", "recall", "optimal_threshold"]
    slim = df[[c for c in cols if c in df.columns]]
    slim.to_csv(PROJECT_ROOT / "reports" / "anomaly_detection_summary.csv", index=False)
    df.to_csv(PROJECT_ROOT / "reports" / "anomaly_detection_summary_full.csv", index=False)
    print(f"\n  ✅ Rebuilt anomaly_detection_summary.csv ({len(rows)} categories)")
    return df


def plot_master_dashboard(summary_df, save_dir):
    cats   = summary_df["category"].tolist()
    n      = len(cats)
    idx    = np.arange(n)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.95, n))

    def save_bar(values, ylabel, title, fname, color_line, mean_label):
        fig, ax = plt.subplots(figsize=(max(10, n*0.9), 5))
        bars = ax.bar(idx, values*100, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(idx); ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
        ax.set_ylim([0, 110]); ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axhline(values.mean()*100, color=color_line, linestyle="-.", lw=1.5,
                   label=f"{mean_label} Mean: {values.mean()*100:.1f}%")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{v*100:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir/fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✔ {fname}")

    save_bar(summary_df["auroc"],    "AUROC (%)",    "AUROC — All Categories",    "dashboard_auroc.png",    "#E74C3C", "AUROC")
    save_bar(summary_df["auprc"],    "AUPRC (%)",    "AUPRC — All Categories",    "dashboard_auprc.png",    "#2980B9", "AUPRC")
    save_bar(summary_df["f1_score"], "F1-Score (%)", "F1-Score — All Categories", "dashboard_f1.png",       "#27AE60", "F1")

    # Precision vs Recall scatter
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(summary_df["recall"]*100, summary_df["precision"]*100,
                    c=summary_df["f1_score"], cmap="RdYlGn", s=180, edgecolors="k", lw=0.5, zorder=3)
    for _, row in summary_df.iterrows():
        ax.annotate(row["category"], (row["recall"]*100, row["precision"]*100),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    fig.colorbar(sc, ax=ax, label="F1-Score")
    ax.set_xlabel("Recall (%)", fontsize=12); ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("Precision vs Recall (colour = F1)", fontsize=13, fontweight="bold")
    ax.set_xlim([0, 110]); ax.set_ylim([0, 110]); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir/"dashboard_prec_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ dashboard_prec_recall.png")

    # Combined line chart
    fig, ax = plt.subplots(figsize=(max(13, n*0.95), 5))
    ax.plot(idx, summary_df["auroc"]*100,    "o-", lw=2,   color="#E74C3C", label="AUROC")
    ax.plot(idx, summary_df["auprc"]*100,    "s-", lw=2,   color="#2980B9", label="AUPRC")
    ax.plot(idx, summary_df["f1_score"]*100, "^-", lw=2,   color="#27AE60", label="F1-Score")
    ax.plot(idx, summary_df["precision"]*100,"D-", lw=1.5, color="#8E44AD", alpha=0.8, label="Precision")
    ax.plot(idx, summary_df["recall"]*100,   "P-", lw=1.5, color="#F39C12", alpha=0.8, label="Recall")
    ax.set_xticks(idx); ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax.set_ylim([0, 110]); ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Full Performance Comparison — All Categories", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=5); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir/"dashboard_all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ dashboard_all_metrics.png")

    # Confusion matrices
    if all(c in summary_df.columns for c in ["true_positives","true_negatives","false_positives","false_negatives"]):
        ncols = 5; nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5))
        axes = axes.flatten()
        for i, (ax, cat) in enumerate(zip(axes, cats)):
            row = summary_df[summary_df["category"]==cat].iloc[0]
            cm = np.array([[row["true_negatives"], row["false_positives"]],
                           [row["false_negatives"], row["true_positives"]]])
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Pred Normal","Pred Defect"], fontsize=8)
            ax.set_yticklabels(["True Normal","True Defect"], fontsize=8)
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, str(cm[r,c]), ha="center", va="center", fontsize=14,
                            fontweight="bold", color="white" if cm[r,c]>cm.max()*0.6 else "black")
            ax.set_title(cat.title(), fontsize=10, fontweight="bold")
        for ax in axes[n:]: ax.axis("off")
        plt.suptitle("Confusion Matrices — All Categories", fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_dir/"dashboard_confusion_matrices.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✔ dashboard_confusion_matrices.png")


def main():
    # Categories to train — from CLI args or default remaining ones
    if len(sys.argv) > 1:
        categories = [c.strip() for c in sys.argv[1:] if c.strip()]
    else:
        # Default: all 15 categories
        categories = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
                      "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]

    print("=" * 65)
    print(f"  DEFECTVISION — TRAINING: {', '.join(categories)}")
    print("=" * 65)
    print(f"  Device: {device_str.upper()} | Coreset: {SUBSAMPLE_RATIO*100:.0f}%")
    sys.stdout.flush()

    if not TORCH_AVAILABLE:
        print("❌ PyTorch not found."); sys.exit(1)

    from src.feature_extraction import DeepFeatureExtractor
    extractor = DeepFeatureExtractor(backbone_name="resnet50", device=device_str)

    dataset_root = PROJECT_ROOT / "data" / "mvtec"
    metadata_path = PROJECT_ROOT / "data" / "embeddings" / "metadata.csv"

    metadata_df = pd.read_csv(metadata_path)
    # Fix NaN mask_path → None
    metadata_df["mask_path"] = metadata_df["mask_path"].where(
        metadata_df["mask_path"].notna(), other=None
    )
    print(f"  Metadata loaded: {len(metadata_df):,} images\n")
    sys.stdout.flush()

    all_metrics = []
    failed = []
    t_global = time.time()

    for cat in categories:
        try:
            m = run_category(cat, metadata_df, extractor)
            if m:
                all_metrics.append(m)
        except Exception as e:
            import traceback
            print(f"\n  ❌ FAILED for {cat}: {e}")
            traceback.print_exc()
            failed.append(cat)
        sys.stdout.flush()

    total_time = time.time() - t_global
    print(f"\n  Total time: {total_time/60:.1f} min")

    # Rebuild full summary from all JSON files
    print("\n  Rebuilding summary CSV from all category metrics …")
    summary_df = rebuild_summary()

    if summary_df is not None and len(summary_df) > 0:
        print("\n  Generating dashboard charts …")
        plot_master_dashboard(summary_df, PROJECT_ROOT / "results" / "plots")

        print("\n" + "=" * 65)
        print("  🏆  SCOREBOARD")
        print("=" * 65)
        display_cols = ["category","auroc","auprc","f1_score","precision","recall"]
        avail = [c for c in display_cols if c in summary_df.columns]
        print(summary_df[avail].to_string(index=False))
        print(f"\n  Mean AUROC : {summary_df['auroc'].mean()*100:.2f}%")
        print(f"  Mean AUPRC : {summary_df['auprc'].mean()*100:.2f}%")
        print(f"  Mean F1    : {summary_df['f1_score'].mean()*100:.2f}%")

    if failed:
        print(f"\n  ⚠ Failed: {', '.join(failed)}")
    print(f"\n  ✅ Models  → {PROJECT_ROOT/'models'}")
    print(f"  ✅ Reports → {PROJECT_ROOT/'reports'}")
    print(f"  ✅ Plots   → {PROJECT_ROOT/'results'/'plots'}")
    print("  Done!\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
