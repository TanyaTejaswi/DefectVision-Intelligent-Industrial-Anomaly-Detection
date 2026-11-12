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
