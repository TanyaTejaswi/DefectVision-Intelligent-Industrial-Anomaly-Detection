"""
Preprocessing Module
Handles dataset parsing, metadata compilation, and standard image transformations
for MVTec AD industrial categories (bottle, cable, hazelnut, etc.).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Fallback base class


# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transforms(img_size: int = 224, crop_size: int = 224):
    """
    Returns standard torchvision preprocessing transforms.
    Resize -> CenterCrop -> ToTensor -> Normalize (ImageNet stats).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and Torchvision are required for torchvision transforms.")

    return T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def build_mvtec_metadata(
    dataset_root: Union[str, Path],
    categories: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Scans the MVTec AD directory and constructs a structured metadata DataFrame.

    Columns produced:
    - image_id: unique string identifier
    - category: e.g. bottle, cable, hazelnut
    - split: 'train' or 'test'
    - defect_type: e.g. 'good', 'broken_large', 'bent_wire'
    - is_anomaly: 0 for good (normal), 1 for defective
    - image_path: absolute or relative path to raw image
    - mask_path: path to ground-truth binary mask (if defective test sample), else None
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root '{dataset_root}' does not exist.")

    if categories is None:
        categories = [d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]

    records = []

    for cat in sorted(categories):
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue

        # 1. Process 'train' split (normally only 'good' subfolder)
        train_dir = cat_dir / "train"
        if train_dir.is_dir():
            for defect_dir in sorted(train_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_name = defect_dir.name
                for img_file in sorted(defect_dir.glob("*.png")):
                    records.append({
                        "image_id": f"{cat}_train_{defect_name}_{img_file.stem}",
                        "category": cat,
                        "split": "train",
                        "defect_type": defect_name,
                        "is_anomaly": 0 if defect_name == "good" else 1,
                        "image_path": str(img_file.resolve()),
                        "mask_path": None
                    })

        # 2. Process 'test' split (both 'good' and defect folders)
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        if test_dir.is_dir():
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_name = defect_dir.name
                is_anomaly = 0 if defect_name == "good" else 1

                for img_file in sorted(defect_dir.glob("*.png")):
                    mask_path = None
                    if is_anomaly and gt_dir.is_dir():
                        expected_mask = gt_dir / defect_name / f"{img_file.stem}_mask.png"
                        if expected_mask.exists():
                            mask_path = str(expected_mask.resolve())

                    records.append({
                        "image_id": f"{cat}_test_{defect_name}_{img_file.stem}",
                        "category": cat,
                        "split": "test",
                        "defect_type": defect_name,
                        "is_anomaly": is_anomaly,
                        "image_path": str(img_file.resolve()),
                        "mask_path": mask_path
                    })

    df = pd.DataFrame(
        records,
        columns=["image_id", "category", "split", "defect_type", "is_anomaly", "image_path", "mask_path"]
    )
    return df


def build_industrial_suite_metadata(
    project_root: Union[str, Path] = ".",
    categories: Optional[List[str]] = None,
    max_samples_per_category: Optional[int] = None
) -> pd.DataFrame:
    """
    Builds a unified, authoritative metadata DataFrame covering the complete
    8-Line Industrial Quality Suite across MVTec AD, Amazon VisA, and NEU Steel:
    1. pcb1 (Amazon VisA)
    2. pcb2 (Amazon VisA)
    3. pcb3 (Amazon VisA)
    4. pcb4 (Amazon VisA)
    5. capsules (Amazon VisA)
    6. bottle (MVTec AD)
    7. cable (MVTec AD)
    8. steel (NEU Surface Defect)
    """
    root = Path(project_root).resolve()
    if categories is None:
        categories = ["pcb1", "pcb2", "pcb3", "pcb4", "capsules", "bottle", "cable", "steel"]

    records = []

    # 1. VisA Categories
    visa_cats = [c for c in categories if c in ["pcb1", "pcb2", "pcb3", "pcb4", "capsules"]]
    visa_dir = root / "datasets" / "visa"
    if visa_dir.exists():
        split_csv = visa_dir / "split_csv" / "1cls.csv"
        if split_csv.exists():
            visa_split_df = pd.read_csv(split_csv)
            for cat in visa_cats:
                cat_sub = visa_split_df[visa_split_df["object"] == cat]
                for _, row in cat_sub.iterrows():
                    img_rel = str(row["image"]).replace("\\", "/")
                    full_img = visa_dir / img_rel
                    if not full_img.exists():
                        continue
                    is_anom = 1 if row["label"] == "anomaly" else 0
                    mask_p = None
                    if is_anom and pd.notna(row.get("mask")):
                        full_mask = visa_dir / str(row["mask"]).replace("\\", "/")
                        if full_mask.exists():
                            mask_p = str(full_mask.resolve())

                    records.append({
                        "image_id": f"{cat}_{row['split']}_{row['label']}_{full_img.stem}",
                        "category": cat,
                        "split": str(row["split"]),
                        "defect_type": "defect" if is_anom else "good",
                        "is_anomaly": is_anom,
                        "image_path": str(full_img.resolve()),
                        "mask_path": mask_p
                    })
        else:
            # Direct folder scanning fallback
            for cat in visa_cats:
                cat_dir = visa_dir / cat / "Data" / "Images"
                norm_files = sorted(list((cat_dir / "Normal").glob("*.*"))) if (cat_dir / "Normal").exists() else []
                anom_files = sorted(list((cat_dir / "Anomaly").glob("*.*"))) if (cat_dir / "Anomaly").exists() else []
                # Train: 80% normal, Test: 20% normal + all anomaly
                n_split = int(len(norm_files) * 0.8)
                for f in norm_files[:n_split]:
                    records.append({
                        "image_id": f"{cat}_train_good_{f.stem}",
                        "category": cat,
                        "split": "train",
                        "defect_type": "good",
                        "is_anomaly": 0,
                        "image_path": str(f.resolve()),
                        "mask_path": None
                    })
                for f in norm_files[n_split:]:
                    records.append({
                        "image_id": f"{cat}_test_good_{f.stem}",
                        "category": cat,
                        "split": "test",
                        "defect_type": "good",
                        "is_anomaly": 0,
                        "image_path": str(f.resolve()),
                        "mask_path": None
                    })
                for f in anom_files:
                    records.append({
                        "image_id": f"{cat}_test_defect_{f.stem}",
                        "category": cat,
                        "split": "test",
                        "defect_type": "defect",
                        "is_anomaly": 1,
                        "image_path": str(f.resolve()),
                        "mask_path": None
                    })

    # 2. MVTec Categories
    mvtec_cats = [c for c in categories if c in ["bottle", "cable"]]
    mvtec_dir = root / "data" / "mvtec"
    if mvtec_dir.exists() and mvtec_cats:
        mvtec_df = build_mvtec_metadata(mvtec_dir, categories=mvtec_cats)
        records.extend(mvtec_df.to_dict(orient="records"))

    # 3. Steel (NEU) Category
    if "steel" in categories:
        steel_dir = root / "data" / "steel"
        if steel_dir.exists():
            tg = steel_dir / "train" / "good"
            if tg.exists():
                for f in sorted(list(tg.glob("*.png"))):
                    records.append({
                        "image_id": f"steel_train_good_{f.stem}",
                        "category": "steel",
                        "split": "train",
                        "defect_type": "good",
                        "is_anomaly": 0,
                        "image_path": str(f.resolve()),
                        "mask_path": None
                    })
            test_dir = steel_dir / "test"
            if test_dir.exists():
                for sub in sorted(list(test_dir.iterdir())):
                    if not sub.is_dir():
                        continue
                    is_anom = 0 if sub.name == "good" else 1
                    for f in sorted(list(sub.glob("*.png"))):
                        records.append({
                            "image_id": f"steel_test_{sub.name}_{f.stem}",
                            "category": "steel",
                            "split": "test",
                            "defect_type": sub.name,
                            "is_anomaly": is_anom,
                            "image_path": str(f.resolve()),
                            "mask_path": None
                        })

    df = pd.DataFrame(
        records,
        columns=["image_id", "category", "split", "defect_type", "is_anomaly", "image_path", "mask_path"]
    )
    return df


class MVTecDataset(Dataset):
    """
    PyTorch Dataset implementation for MVTec Anomaly Detection.
    """
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform=None,
        mask_transform=None,
        target_split: Optional[str] = None,
        target_category: Optional[str] = None
    ):
        df = metadata_df.copy()
        if target_split is not None:
            df = df[df["split"] == target_split]
        if target_category is not None:
            df = df[df["category"] == target_category]

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        mask_path = row["mask_path"]

        # Load RGB image
        img = Image.open(image_path).convert("RGB")
        orig_size = img.size  # (W, H)

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = np.array(img)

        # Load ground truth mask if available
        mask_tensor = None
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform is not None:
                mask_tensor = self.mask_transform(mask)
            elif TORCH_AVAILABLE:
                # Default binarization: Resize and convert to tensor (0.0 or 1.0)
                mask_tensor = T.functional.to_tensor(mask.resize((224, 224), Image.NEAREST))
                mask_tensor = (mask_tensor > 0.5).float()
            else:
                mask_tensor = np.array(mask) > 128
        else:
            if TORCH_AVAILABLE:
                mask_tensor = torch.zeros((1, 224, 224), dtype=torch.float32)
            else:
                mask_tensor = np.zeros((224, 224), dtype=np.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "is_anomaly": int(row["is_anomaly"]),
            "category": row["category"],
            "defect_type": row["defect_type"],
            "split": row["split"],
            "image_id": row["image_id"],
            "image_path": image_path,
            "orig_size": orig_size
        }


if __name__ == "__main__":
    # Test metadata builder
    sample_root = Path("data/mvtec")
    if sample_root.exists():
        df = build_mvtec_metadata(sample_root, categories=["bottle", "cable"])
        print(f"Metadata created with {len(df)} total samples.")
        print(df.groupby(["category", "split", "is_anomaly"]).size())
