"""
Custom Training Script for PCB1..PCB4, Steel, and Capsules in DefectVision.
Uses ResNet-50 PatchCore feature extraction to fit MemoryBankAnomalyDetector models,
evaluates AUROC / F1, and saves production memory bank models.
"""

import sys
import json
import time
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, precision_score,
    recall_score, f1_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.feature_extraction import DeepFeatureExtractor
from src.preprocessing import get_default_transforms
from src.anomaly_detection import MemoryBankAnomalyDetector

def evaluate_metrics(y_true, y_scores, category_name="unknown"):
    auroc = float(roc_auc_score(y_true, y_scores))
    auprc = float(average_precision_score(y_true, y_scores))
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls), 0.0
    )
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) > 0 else 0
    optimal_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else float(np.mean(y_scores))
    y_pred = (y_scores >= optimal_threshold).astype(int)
    prec  = float(precision_score(y_true, y_pred, zero_division=0))
    rec   = float(recall_score(y_true, y_pred, zero_division=0))
    f1    = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "category": category_name,
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

def train_and_eval_category(
    category: str,
    train_normal_paths: list,
    test_normal_paths: list,
    test_defect_paths: list,
    extractor: DeepFeatureExtractor,
    transforms,
    subsample_ratio: float = 0.05
):
    print(f"\n{'='*60}")
    print(f"  Training PatchCore for: {category.upper()}")
    print(f"  Train Normals: {len(train_normal_paths)} | Test Normals: {len(test_normal_paths)} | Test Defects: {len(test_defect_paths)}")
    print(f"{'='*60}")

    t0 = time.time()
    train_patches = []
    for i, p in enumerate(train_normal_paths):
        img = Image.open(p).convert("RGB")
        tensor = transforms(img)
        patches = extractor.extract_patch_features(tensor, (28, 28))
        p_np = patches.squeeze(0).permute(1, 2, 0).reshape(-1, 1536).cpu().numpy()
        train_patches.append(p_np)
        if (i + 1) % 20 == 0 or (i + 1) == len(train_normal_paths):
            print(f"    Train progress: {i+1}/{len(train_normal_paths)} ({time.time()-t0:.1f}s)")

    all_train = np.vstack(train_patches)
    print(f"  Total patches: {all_train.shape[0]:,} -> fitting memory bank (subsample={subsample_ratio}) ...")

    detector = MemoryBankAnomalyDetector(
        category=category,
        n_neighbors=1,
        subsample_ratio=subsample_ratio,
        gaussian_sigma=4.0,
        random_state=42
    )
    detector.fit(all_train)
    print(f"  Fitted Memory Bank vectors: {detector.memory_bank.shape[0]:,}")

    # Evaluate on Test Normal and Test Defect
    y_true = []
    y_scores = []
    print("  Evaluating test set ...")
    for p in test_normal_paths:
        img = Image.open(p).convert("RGB")
        tensor = transforms(img)
        patches = extractor.extract_patch_features(tensor, (28, 28))
        p_np = patches.squeeze(0).permute(1, 2, 0).reshape(-1, 1536).cpu().numpy()
        score, _ = detector.predict_image_and_map(p_np, (28, 28), (224, 224))
        y_true.append(0)
        y_scores.append(score)

    for p in test_defect_paths:
        img = Image.open(p).convert("RGB")
        tensor = transforms(img)
        patches = extractor.extract_patch_features(tensor, (28, 28))
        p_np = patches.squeeze(0).permute(1, 2, 0).reshape(-1, 1536).cpu().numpy()
        score, _ = detector.predict_image_and_map(p_np, (28, 28), (224, 224))
        y_true.append(1)
        y_scores.append(score)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    norm_scores = y_scores[y_true == 0]
    def_scores = y_scores[y_true == 1]
    print(f"  Normal test scores: min={norm_scores.min():.2f}, mean={norm_scores.mean():.2f}, max={norm_scores.max():.2f}")
    print(f"  Defect test scores: min={def_scores.min():.2f}, mean={def_scores.mean():.2f}, max={def_scores.max():.2f}")

    metrics = evaluate_metrics(y_true, y_scores, category_name=category)
    detector.optimal_threshold = metrics["optimal_threshold"]
    print(f"  Results for {category.upper()}: AUROC={metrics['auroc']*100:.2f}%, F1={metrics['f1_score']*100:.2f}%, Threshold={metrics['optimal_threshold']:.4f}")

    # Save detector
    out_path = PROJECT_ROOT / f"models/memory_bank_{category}.pkl"
    detector.save(out_path)
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Save metrics JSON
    with open(PROJECT_ROOT / f"reports/{category}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def main():
    extractor = DeepFeatureExtractor("resnet50")
    transforms = get_default_transforms(224, 224)

    # 1. Sync capsule -> capsules
    cap_src = PROJECT_ROOT / "models/memory_bank_capsule.pkl"
    cap_dst = PROJECT_ROOT / "models/memory_bank_capsules.pkl"
    if cap_src.exists():
        shutil.copy2(cap_src, cap_dst)
        print(f"[OK] Synced {cap_src.name} -> {cap_dst.name} ({cap_dst.stat().st_size/1e6:.1f} MB)")

    all_metrics = {}

    # 2. Train PCB1..PCB4
    for pcb in ["pcb1", "pcb2", "pcb3", "pcb4"]:
        norm_dir = PROJECT_ROOT / f"datasets/visa/{pcb}/Data/Images/Normal"
        anom_dir = PROJECT_ROOT / f"datasets/visa/{pcb}/Data/Images/Anomaly"
        if not norm_dir.exists() or not anom_dir.exists():
            print(f"Skipping {pcb}, directory not found.")
            continue

        all_norm = sorted(list(norm_dir.glob("*.JPG")) + list(norm_dir.glob("*.jpg")) + list(norm_dir.glob("*.png")))
        all_anom = sorted(list(anom_dir.glob("*.JPG")) + list(anom_dir.glob("*.jpg")) + list(anom_dir.glob("*.png")))

        # Use 60 normals for train, 40 normals for test, all anomaly for test
        train_norm = all_norm[:60]
        test_norm = all_norm[60:100]
        test_anom = all_anom

        metrics = train_and_eval_category(
            category=pcb,
            train_normal_paths=train_norm,
            test_normal_paths=test_norm,
            test_defect_paths=test_anom,
            extractor=extractor,
            transforms=transforms,
            subsample_ratio=0.08
        )
        all_metrics[pcb] = metrics

    # 3. Train Steel
    steel_train_good = list((PROJECT_ROOT / "data/steel/train/good").glob("*.png"))
    steel_test_good = list((PROJECT_ROOT / "data/steel/test/good").glob("*.png"))
    steel_defects = []
    for defect_folder in (PROJECT_ROOT / "data/steel/test").iterdir():
        if defect_folder.is_dir() and defect_folder.name != "good":
            steel_defects.extend(list(defect_folder.glob("*.png")))

    if steel_train_good and steel_defects:
        metrics = train_and_eval_category(
            category="steel",
            train_normal_paths=steel_train_good + steel_test_good[:5],
            test_normal_paths=steel_test_good[5:],
            test_defect_paths=steel_defects,
            extractor=extractor,
            transforms=transforms,
            subsample_ratio=0.1
        )
        all_metrics["steel"] = metrics

    print("\n" + "="*60)
    print("TRAINING SUMMARY:")
    for k, v in all_metrics.items():
        print(f"  {k:8s} | AUROC: {v['auroc']*100:6.2f}% | F1: {v['f1_score']*100:6.2f}% | Threshold: {v['optimal_threshold']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
