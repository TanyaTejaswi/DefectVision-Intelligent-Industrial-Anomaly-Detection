"""
Tanya's Anomaly Detection Master Pipeline
Executes end-to-end PatchCore anomaly detection across all 8 target categories:
- pcb1, pcb2, pcb3, pcb4 (Amazon VisA)
- capsules (Amazon VisA)
- bottle, cable (MVTec AD)
- steel (NEU Surface Defect)

Produces:
1. Serialized Memory Bank Models: models/memory_bank_{category}.pkl
2. Academic Evaluation Metrics: reports/anomaly_detection_summary.csv & reports/{category}_metrics.json
3. Evaluation Curves: results/plots/{category}_roc_pr.png
4. Inspection Overlays: results/plots/{category}_sample_0.png
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

from src.preprocessing import build_industrial_suite_metadata, build_mvtec_metadata, MVTecDataset, get_default_transforms
from src.anomaly_detection import MemoryBankAnomalyDetector, AnomalyInferenceEngine
from src.visualization import plot_anomaly_comparison, overlay_heatmap_on_image
from src.evaluation import evaluate_image_level, plot_roc_pr_curves
from scripts.train_industrial_suite import extract_image_patches, train_and_eval_category


def main():
    print("=" * 70)
    print("DEFECTVISION: MULTI-CATEGORY INDUSTRIAL ANOMALY DETECTION ENGINE")
    print("8-Line Suite: pcb1, pcb2, pcb3, pcb4, capsules, bottle, cable, steel")
    print("=" * 70)

    metadata_path = PROJECT_ROOT / "data" / "embeddings" / "metadata_full.csv"
    if not metadata_path.exists():
        metadata_path = PROJECT_ROOT / "data" / "embeddings" / "metadata.csv"

    if not metadata_path.exists():
        print("Building metadata index for the complete 8-line suite...")
        metadata_df = build_industrial_suite_metadata(PROJECT_ROOT)
        metadata_df.to_csv(metadata_path, index=False)
    else:
        print(f"Loading metadata from {metadata_path}...")
        metadata_df = pd.read_csv(metadata_path)

    categories = ["pcb1", "pcb2", "pcb3", "pcb4", "capsules", "bottle", "cable", "steel"]
    summary_rows = []

    for cat in categories:
        metrics = train_and_eval_category(
            category=cat,
            df=metadata_df,
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
    print(f"✅ All 8 memory bank models saved in: {(PROJECT_ROOT / 'models').resolve()}")


if __name__ == "__main__":
    main()

