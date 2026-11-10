"""
Tanya's Anomaly Detection Master Pipeline
Executes end-to-end PatchCore anomaly detection across all target categories:
- bottle (292 images)
- cable (374 images)
- hazelnut (501 images)

Produces:
1. Serialized Memory Bank Models: models/memory_bank_{category}.pkl
2. Academic Evaluation Metrics: reports/anomaly_detection_summary.csv & reports/{category}_metrics.json
3. Evaluation Curves: results/plots/{category}_roc_pr.png
4. Inspection Overlays: results/plots/{category}_sample.png
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure directories exist
(PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "results" / "plots").mkdir(parents=True, exist_ok=True)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch is not installed in this environment.")
    print("👉 For full GPU/ResNet-50 patch extraction, run this script inside your Google Colab runtime.")

from src.preprocessing import build_mvtec_metadata, MVTecDataset, get_default_transforms
from src.anomaly_detection import MemoryBankAnomalyDetector, AnomalyInferenceEngine
from src.visualization import plot_anomaly_comparison, overlay_heatmap_on_image
from src.evaluation import evaluate_image_level, plot_roc_pr_curves


def run_category_anomaly_detection(
    category: str,
    metadata_df: pd.DataFrame,
    extractor=None,
    device: str = "cpu",
    subsample_ratio: float = 0.1
) -> Dict:
    """
    Runs complete training & evaluation for a single category.
    """
    print(f"\n{'='*60}")
    print(f"🚀 Running PatchCore Anomaly Detection on: {category.upper()}")
    print(f"{'='*60}")

    transforms = get_default_transforms(img_size=224, crop_size=224)
    train_dataset = MVTecDataset(metadata_df, transform=transforms, target_split="train", target_category=category)
    test_dataset = MVTecDataset(metadata_df, transform=transforms, target_split="test", target_category=category)

    print(f"Dataset split: {len(train_dataset)} normal train images | {len(test_dataset)} test images")

    # 1. Extract patch features from normal training images
    print(f"Extracting multi-scale patch features (Layer2+Layer3) from normal training set...")
    train_patches = []

    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        img = item["image"]
        patches = extractor.extract_patch_features(img, target_size=(28, 28))
        # Shape: (B=1, C=1536, H=28, W=28) -> (784, 1536)
        p_np = patches.squeeze(0).permute(1, 2, 0).reshape(-1, patches.shape[1]).cpu().numpy()
        train_patches.append(p_np)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(train_dataset):
            print(f"  Processed {idx + 1}/{len(train_dataset)} train images")

    all_train_patches = np.vstack(train_patches)
    print(f"Total normal patches collected: {all_train_patches.shape[0]} (dim={all_train_patches.shape[1]})")

    # 2. Build and fit Memory Bank
    detector = MemoryBankAnomalyDetector(
        category=category,
        n_neighbors=1,
        subsample_ratio=subsample_ratio,
        gaussian_sigma=4.0
    )
    detector.fit(all_train_patches)

    # 3. Score all test images
    print(f"Scoring {len(test_dataset)} test images and generating spatial anomaly heatmaps...")
    y_true = []
    y_scores = []
    sample_comparisons = []

    for idx in range(len(test_dataset)):
        item = test_dataset[idx]
        img = item["image"]
        is_anomaly = item["is_anomaly"]
        defect_type = item["defect_type"]

        patches = extractor.extract_patch_features(img, target_size=(28, 28))
        p_np = patches.squeeze(0).permute(1, 2, 0).reshape(-1, patches.shape[1]).cpu().numpy()

        score, heat_map = detector.predict_image_and_map(p_np, spatial_grid=(28, 28), target_size=(224, 224))
        y_true.append(is_anomaly)
        y_scores.append(score)

        # Collect 1 normal and 2 defect samples for visual inspection plot
        if (is_anomaly == 0 and len([s for s in sample_comparisons if s["is_anomaly"] == 0]) < 1) or \
           (is_anomaly == 1 and len([s for s in sample_comparisons if s["is_anomaly"] == 1]) < 2):
            raw_img = Image.open(item["image_path"]).convert("RGB").resize((224, 224))
            mask_np = item["mask"].squeeze(0).cpu().numpy() if isinstance(item["mask"], torch.Tensor) else item["mask"]
            sample_comparisons.append({
                "image": raw_img,
                "mask": mask_np,
                "heatmap": heat_map,
                "score": score,
                "defect_type": defect_type,
                "is_anomaly": is_anomaly
            })

    # 4. Evaluate metrics
    metrics = evaluate_image_level(np.array(y_true), np.array(y_scores), category_name=category)
    detector.optimal_threshold = metrics["optimal_threshold"]

    print(f"\n--- Benchmark Results for {category.upper()} ---")
    print(f"  AUROC:             {metrics['auroc'] * 100:.2f}%")
    print(f"  AUPRC:             {metrics['auprc'] * 100:.2f}%")
    print(f"  F1-Score:          {metrics['f1_score']:.4f}")
    print(f"  Precision:         {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:            {metrics['recall'] * 100:.2f}%")
    print(f"  Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"  Confusion Matrix:  TP={metrics['true_positives']} | TN={metrics['true_negatives']} | FP={metrics['false_positives']} | FN={metrics['false_negatives']}")

    # 5. Serialize memory bank model (.pkl)
    model_save_path = PROJECT_ROOT / "models" / f"memory_bank_{category}.pkl"
    detector.save(model_save_path)

    # 6. Save JSON metrics
    metrics_json_path = PROJECT_ROOT / "reports" / f"{category}_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metric report to {metrics_json_path}")

    # 7. Plot ROC & PR curves
    plot_roc_pr_curves(
        np.array(y_true),
        np.array(y_scores),
        category_name=category.title(),
        save_path=PROJECT_ROOT / "results" / "plots" / f"{category}_roc_pr.png"
    )

    # 8. Plot sample inspections
    for idx, comp in enumerate(sample_comparisons):
        plot_anomaly_comparison(
            comp["image"],
            comp["mask"],
            comp["heatmap"],
            comp["score"],
            defect_type=comp["defect_type"],
            category=category,
            save_path=PROJECT_ROOT / "results" / "plots" / f"{category}_sample_{idx}.png"
        )

    return metrics


def main():
    print("=" * 70)
    print("DEFECTVISION: TANYA'S MULTI-CATEGORY ANOMALY DETECTION ENGINE")
    print("Categories: bottle, cable, hazelnut")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("❌ PyTorch is required to run deep feature extraction.")
        print("💡 TIP: Run this script inside your Google Colab session:")
        print("   !python run_tanya_pipeline.py")
        return

    from src.feature_extraction import DeepFeatureExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Device: {device.upper()}")

    extractor = DeepFeatureExtractor(backbone_name="resnet50", device=device)

    # Check dataset
    dataset_root = PROJECT_ROOT / "data" / "mvtec"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist!")

    metadata_path = PROJECT_ROOT / "data" / "embeddings" / "metadata.csv"
    if metadata_path.exists():
        print(f"Loading existing metadata from {metadata_path}...")
        metadata_df = pd.read_csv(metadata_path)
    else:
        print("Building metadata index from data/mvtec...")
        metadata_df = build_mvtec_metadata(dataset_root, categories=["bottle", "cable", "hazelnut"])
        metadata_df.to_csv(metadata_path, index=False)

    categories = ["bottle", "cable", "hazelnut"]
    summary_rows = []

    for cat in categories:
        metrics = run_category_anomaly_detection(
            category=cat,
            metadata_df=metadata_df,
            extractor=extractor,
            device=device,
            subsample_ratio=0.1
        )
        summary_rows.append(metrics)

    # Compile master summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = PROJECT_ROOT / "reports" / "anomaly_detection_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 70)
    print("🏆 MULTI-CATEGORY INDUSTRIAL ANOMALY DETECTION SCOREBOARD")
    print("=" * 70)
    print(summary_df[["category", "auroc", "auprc", "f1_score", "precision", "recall", "optimal_threshold"]].to_string(index=False))
    print(f"\n✅ Summary saved to: {summary_csv_path.resolve()}")
    print(f"✅ All 3 memory bank models saved in: {(PROJECT_ROOT / 'models').resolve()}")


if __name__ == "__main__":
    main()

