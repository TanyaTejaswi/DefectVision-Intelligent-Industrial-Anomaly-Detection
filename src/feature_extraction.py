
"""
Feature Extraction Module

Supports two complementary feature-extraction pipelines:

1. Global image embeddings:
   - ResNet50 feature extraction
   - Image scanning and preprocessing
   - Feature, label, and path export
   - PCA, t-SNE, and clustering workflows

2. DeepFeatureExtractor:
   - ResNet50 and ResNet18 backbones
   - Global image embeddings
   - Multi-scale spatial patch features
   - PatchCore-style anomaly detection workflows
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, ResNet18_Weights


# ---------------------------------------------------------
# DEVICE AND PREPROCESSING UTILITIES
# ---------------------------------------------------------

def get_device(preferred: Optional[str] = None) -> torch.device:
    """Select CUDA when available, otherwise CPU."""

    if preferred is not None:
        return torch.device(preferred)

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def get_image_transform(image_size: int = 224) -> transforms.Compose:
    """Standard ImageNet preprocessing for pretrained ResNet models."""

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ---------------------------------------------------------
# ORIGINAL RESNET50 FEATURE EXTRACTION PIPELINE
# ---------------------------------------------------------

def build_resnet50_extractor(
    device: Optional[torch.device] = None,
    pretrained: bool = True
) -> nn.Module:
    """Build a frozen ResNet50 model without its classification head."""

    if device is None:
        device = get_device()

    weights = (
        models.ResNet50_Weights.DEFAULT
        if pretrained else None
    )

    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model.to(device)

    return model


def extract_batch_features(
    images_tensor: torch.Tensor,
    model: nn.Module,
    device: torch.device
) -> np.ndarray:
    """Extract 2048-dimensional features from a batch of images."""

    model.eval()

    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        features = model(images_tensor)
        features = features.reshape(features.size(0), -1)

    return features.cpu().numpy()


def extract_features_from_image_paths(
    image_paths: List[str],
    model: Optional[nn.Module] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Extract ResNet50 embeddings from image file paths."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if len(image_paths) == 0:
        return np.empty((0, 2048), dtype=np.float32)

    if device is None:
        device = get_device()

    if model is None:
        model = build_resnet50_extractor(
            device=device,
            pretrained=True
        )

    model.eval()
    transform = get_image_transform()

    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            with Image.open(path) as img:
                tensor = transform(img.convert("RGB"))
                batch_tensors.append(tensor)

        stacked_tensor = torch.stack(batch_tensors)

        batch_features = extract_batch_features(
            stacked_tensor,
            model,
            device
        )

        all_embeddings.append(batch_features)

    return np.vstack(all_embeddings)


def scan_industrial_dataset(
    dataset_dir: str
) -> Tuple[List[str], np.ndarray]:
    """
    Scan an MVTec-style dataset.

    Images inside folders named 'good' receive label 0.
    Images in other folders receive label 1.
    """

    valid_extensions = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff"
    }

    image_paths = []
    labels = []

    for root, _, files in os.walk(dataset_dir):
        dir_name = os.path.basename(root).lower()
        is_normal = dir_name == "good"

        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()

            if ext in valid_extensions:
                image_paths.append(
                    os.path.join(root, filename)
                )

                labels.append(0 if is_normal else 1)

    return image_paths, np.array(labels, dtype=int)


def extract_and_save_dataset_features(
    dataset_dir: str,
    output_dir: str = "datasets",
    batch_size: int = 32
) -> Dict[str, str]:
    """
    Original dataset extraction workflow.

    Saves:
    - features.npy
    - labels.npy
    - image_paths.json

    This function retains its original name and arguments
    for compatibility with existing code and tests.
    """

    os.makedirs(output_dir, exist_ok=True)

    image_paths, labels = scan_industrial_dataset(dataset_dir)

    if len(image_paths) == 0:
        raise ValueError(
            f"No valid image files found in {dataset_dir}"
        )

    print(
        f"[EXTRACTOR] Found {len(image_paths)} images."
    )

    print(
        f"[EXTRACTOR] Normal: {int(np.sum(labels == 0))}, "
        f"Defective: {int(np.sum(labels == 1))}"
    )

    features = extract_features_from_image_paths(
        image_paths,
        batch_size=batch_size
    )

    features_path = os.path.join(
        output_dir, "features.npy"
    )

    labels_path = os.path.join(
        output_dir, "labels.npy"
    )

    paths_file = os.path.join(
        output_dir, "image_paths.json"
    )

    np.save(features_path, features)
    np.save(labels_path, labels)

    with open(paths_file, "w") as f:
        json.dump(image_paths, f, indent=2)

    print(
        f"[EXTRACTOR] Saved features with shape {features.shape}"
    )

    return {
        "features_path": features_path,
        "labels_path": labels_path,
        "image_paths_file": paths_file
    }


# ---------------------------------------------------------
# DEEP FEATURE EXTRACTOR
# ---------------------------------------------------------

class DeepFeatureExtractor:
    """
    Extracts global embeddings and spatial patch features
    from intermediate layers of a CNN backbone.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        device: Optional[str] = None
    ):

        self.device = get_device(device)
        self.backbone_name = backbone_name.lower()

        self._init_backbone(pretrained)

    def _init_backbone(self, pretrained: bool):

        if self.backbone_name == "resnet50":
            weights = (
                ResNet50_Weights.DEFAULT
                if pretrained else None
            )

            backbone = models.resnet50(weights=weights)

        elif self.backbone_name == "resnet18":
            weights = (
                ResNet18_Weights.DEFAULT
                if pretrained else None
            )

            backbone = models.resnet18(weights=weights)

        else:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}"
            )

        backbone.eval()

        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = backbone.to(self.device)

        self.features: Dict[str, torch.Tensor] = {}

        self._register_hooks()

    def _register_hooks(self):

        def hook_fn(name):
            def hook(module, inputs, output):
                self.features[name] = output

            return hook

        if hasattr(self.backbone, "layer2"):
            self.backbone.layer2.register_forward_hook(
                hook_fn("layer2")
            )

        if hasattr(self.backbone, "layer3"):
            self.backbone.layer3.register_forward_hook(
                hook_fn("layer3")
            )

    def extract_image_embedding(
        self,
        x: torch.Tensor
    ) -> np.ndarray:
        """Extract global average-pooled image embeddings."""

        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        self.backbone.eval()

        with torch.no_grad():
            out = self.backbone.conv1(x)
            out = self.backbone.bn1(out)
            out = self.backbone.relu(out)
            out = self.backbone.maxpool(out)

            out = self.backbone.layer1(out)
            out = self.backbone.layer2(out)
            out = self.backbone.layer3(out)
            out = self.backbone.layer4(out)

            pooled = self.backbone.avgpool(out)
            embedding = torch.flatten(pooled, 1)

        return embedding.cpu().numpy()

    def extract_patch_features(
        self,
        x: torch.Tensor,
        target_size: Tuple[int, int] = (28, 28)
    ) -> torch.Tensor:
        """
        Extract multi-scale spatial patch features.

        Returns:
            Tensor with shape:
            (B, combined_channels, target_height, target_width)
        """

        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        self.features.clear()
        self.backbone.eval()

        with torch.no_grad():
            _ = self.backbone(x)

            f_l2 = self.features["layer2"]
            f_l3 = self.features["layer3"]

            f_l2_resized = F.interpolate(
                f_l2,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )

            f_l3_resized = F.interpolate(
                f_l3,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )

            patch_features = torch.cat(
                [f_l2_resized, f_l3_resized],
                dim=1
            )

        return patch_features


# ---------------------------------------------------------
# METADATA-BASED EXTRACTION WORKFLOW
# ---------------------------------------------------------

def extract_and_save_metadata_features(
    metadata_df: pd.DataFrame,
    output_dir: Union[str, Path] = "data/embeddings",
    backbone_name: str = "resnet50",
    batch_size: int = 16,
    device: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Metadata-based global embedding workflow.

    This is Tanya's original metadata_df workflow, now exposed
    under a separate name to avoid overwriting the original
    extract_and_save_dataset_features() function.
    """

    from src.preprocessing import (
        MVTecDataset,
        get_default_transforms
    )

    if metadata_df.empty:
        raise ValueError("Metadata dataframe is empty.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    extractor = DeepFeatureExtractor(
        backbone_name=backbone_name,
        device=device
    )

    image_transforms = get_default_transforms(
        img_size=224,
        crop_size=224
    )

    dataset = MVTecDataset(
        metadata_df=metadata_df,
        transform=image_transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    all_embeddings = []

    print(
        f"Extracting {len(dataset)} samples using "
        f"{backbone_name} on {extractor.device}..."
    )

    for batch_idx, batch in enumerate(dataloader):

        imgs = batch["image"]

        embeddings = extractor.extract_image_embedding(imgs)

        all_embeddings.append(embeddings)

        if (
            (batch_idx + 1) % 10 == 0
            or (batch_idx + 1) == len(dataloader)
        ):
            print(
                f"Processed batch {batch_idx + 1}/"
                f"{len(dataloader)}"
            )

    if not all_embeddings:
        raise ValueError("No embeddings were extracted.")

    embeddings_matrix = np.vstack(all_embeddings)

    npy_path = out_path / "image_embeddings.npy"
    csv_path = out_path / "metadata.csv"

    np.save(npy_path, embeddings_matrix)
    metadata_df.to_csv(csv_path, index=False)

    print(
        f"Saved embeddings with shape "
        f"{embeddings_matrix.shape} to {npy_path}"
    )

    print(f"Saved metadata to {csv_path}")

    return embeddings_matrix, metadata_df
