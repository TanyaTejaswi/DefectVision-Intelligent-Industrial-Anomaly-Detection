"""
Feature Extraction Module
Extracts deep representations from industrial images using pretrained CNNs (ResNet / EfficientNet).
Supports:
1. Global image-level embeddings (for PCA, t-SNE, Clustering - Gayathri's pipeline)
2. Patch-level spatial feature maps (for PatchCore / Memory Bank - Tanya's pipeline)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights, ResNet18_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DeepFeatureExtractor:
    """
    Extracts deep embeddings from intermediate and final layers of a CNN backbone.
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        device: Optional[str] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and Torchvision are required for DeepFeatureExtractor. "
                "Please run on Google Colab or install torch locally."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone_name = backbone_name.lower()
        self._init_backbone(pretrained)

    def _init_backbone(self, pretrained: bool):
        if self.backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        elif self.backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = backbone.to(self.device)

        # Hook storage for intermediate patch feature maps
        self.features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def _hook(module, input, output):
                self.features[name] = output
            return _hook

        # Register forward hooks on intermediate layers for patch extraction
        if hasattr(self.backbone, "layer2"):
            self.backbone.layer2.register_forward_hook(hook_fn("layer2"))
        if hasattr(self.backbone, "layer3"):
            self.backbone.layer3.register_forward_hook(hook_fn("layer3"))

    def extract_image_embedding(self, x: "torch.Tensor") -> np.ndarray:
        """
        Extracts global pooled image embedding vector (e.g. 2048-dim for ResNet50).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            # Pass through stem and conv layers
            out = self.backbone.conv1(x)
            out = self.backbone.bn1(out)
            out = self.backbone.relu(out)
            out = self.backbone.maxpool(out)

            out = self.backbone.layer1(out)
            out = self.backbone.layer2(out)
            out = self.backbone.layer3(out)
            out = self.backbone.layer4(out)

            # Global average pooling
            pooled = self.backbone.avgpool(out)
            embedding = torch.flatten(pooled, 1)

        return embedding.cpu().numpy()

    def extract_patch_features(
        self,
        x: "torch.Tensor",
        target_size: Tuple[int, int] = (28, 28)
    ) -> "torch.Tensor":
        """
        Extracts multi-scale patch features from layer2 & layer3, resized to target_size.
        Used for PatchCore memory bank and pixel-level anomaly localization.
        Returns: Tensor of shape (B, channels, target_size[0], target_size[1])
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        self.features.clear()

        with torch.no_grad():
            _ = self.backbone(x)
            f_l2 = self.features["layer2"]
            f_l3 = self.features["layer3"]

            # Align spatial dimensions using bilinear interpolation
            f_l2_resized = F.interpolate(f_l2, size=target_size, mode="bilinear", align_corners=False)
            f_l3_resized = F.interpolate(f_l3, size=target_size, mode="bilinear", align_corners=False)

            # Concatenate multi-scale representations
            patch_features = torch.cat([f_l2_resized, f_l3_resized], dim=1)

        return patch_features


def extract_and_save_dataset_features(
    metadata_df: pd.DataFrame,
    output_dir: Union[str, Path] = "data/embeddings",
    backbone_name: str = "resnet50",
    batch_size: int = 16,
    device: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extracts global image embeddings for all samples in metadata_df and persists them.
    Deliverable for Gayathri's PCA, t-SNE, and Clustering pipeline.
    """
    from src.preprocessing import MVTecDataset, get_default_transforms

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    extractor = DeepFeatureExtractor(backbone_name=backbone_name, device=device)
    transforms = get_default_transforms(img_size=224, crop_size=224)

    dataset = MVTecDataset(metadata_df=metadata_df, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    print(f"Extracting features for {len(dataset)} samples using {backbone_name} on {extractor.device}...")

    for batch_idx, batch in enumerate(dataloader):
        imgs = batch["image"]
        embeddings = extractor.extract_image_embedding(imgs)
        all_embeddings.append(embeddings)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    embeddings_matrix = np.vstack(all_embeddings)

    # Save array and accompanying metadata
    npy_path = out_path / "image_embeddings.npy"
    csv_path = out_path / "metadata.csv"

    np.save(npy_path, embeddings_matrix)
    metadata_df.to_csv(csv_path, index=False)

    print(f"Saved embeddings matrix of shape {embeddings_matrix.shape} to {npy_path}")
    print(f"Saved metadata index to {csv_path}")

    return embeddings_matrix, metadata_df
