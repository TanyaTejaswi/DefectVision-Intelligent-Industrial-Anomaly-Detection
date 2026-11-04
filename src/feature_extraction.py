"""
Feature Extraction Module
Responsibility: Tanya (Computer Vision & Feature Engineering)

This module handles:
1. Deep convolutional neural network loading (Pre-trained ResNet50 backbone).
2. Industrial image preprocessing pipelines (Resize, CenterCrop, ImageNet Normalization).
3. 2,048-dimensional feature embedding extraction from raw product images.
4. Exporting feature arrays (features.npy) and metadata for clustering & anomaly discovery.
"""

from typing import List, Tuple, Optional, Dict, Any
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Selects the optimal hardware device (CUDA GPU if available, otherwise CPU).

    Args:
        preferred: Optional override ('cuda' or 'cpu').

    Returns:
        device: torch.device instance.
    """
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_transform(image_size: int = 224) -> transforms.Compose:
    """
    Standard PyTorch preprocessing pipeline for ResNet50.

    Args:
        image_size: Target cropped dimension (default: 224x224).

    Returns:
        transform: Composed PyTorch image transformations.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def build_resnet50_extractor(
    device: Optional[torch.device] = None,
    pretrained: bool = True
) -> nn.Module:
    """
    Loads ResNet50, replaces the final classification layer with Identity,
    and freezes all parameters for deterministic feature extraction.

    Args:
        device: Target hardware device (CPU or CUDA).
        pretrained: If True, downloads/loads ImageNet pre-trained weights.

    Returns:
        model: ResNet50 feature extractor outputting 2,048-dimensional vectors.
    """
    if device is None:
        device = get_device()

    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Replace the 1,000-class classification head with an Identity layer
    # so the model outputs the raw 2,048-dim feature representation
    model.fc = nn.Identity()

    # Freeze model weights for pure feature extraction (no gradients)
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
    """
    Runs a batch of preprocessed image tensors through the feature extractor.

    Args:
        images_tensor: Tensor of shape (B, 3, 224, 224).
        model: Fitted feature extractor model.
        device: Hardware device.

    Returns:
        embeddings: Numpy array of shape (B, 2048).
    """
    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        features = model(images_tensor)
        # Flatten if needed
        features = features.view(features.size(0), -1)
        return features.cpu().numpy()


def extract_features_from_image_paths(
    image_paths: List[str],
    model: Optional[nn.Module] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Extracts 2,048-dimensional feature embeddings for a list of image file paths.

    Args:
        image_paths: List of absolute or relative image file paths.
        model: Pre-loaded ResNet50 extractor (initialized automatically if None).
        batch_size: Batch size for forward passes.
        device: Hardware device.

    Returns:
        embeddings: Feature matrix of shape (len(image_paths), 2048).
    """
    if len(image_paths) == 0:
        return np.empty((0, 2048), dtype=np.float32)

    if device is None:
        device = get_device()
    if model is None:
        model = build_resnet50_extractor(device=device, pretrained=True)

    transform = get_image_transform()
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors: List[torch.Tensor] = []

        for p in batch_paths:
            with Image.open(p) as img:
                # Ensure 3-channel RGB
                img_rgb = img.convert("RGB")
                tensor = transform(img_rgb)
                batch_tensors.append(tensor)

        stacked_tensor = torch.stack(batch_tensors, dim=0)
        batch_feats = extract_batch_features(stacked_tensor, model, device)
        all_embeddings.append(batch_feats)

    return np.vstack(all_embeddings)


def scan_industrial_dataset(dataset_dir: str) -> Tuple[List[str], np.ndarray]:
    """
    Scans an industrial inspection dataset directory (e.g. MVTec AD structure)
    and collects valid image paths alongside binary labels:
      - Label 0: Normal / Good
      - Label 1: Defect / Anomaly

    Expected structure example:
      dataset_dir/
        train/good/*.png (Normal: 0)
        test/good/*.png (Normal: 0)
        test/broken_large/*.png (Defect: 1)
        test/contamination/*.png (Defect: 1)

    Args:
        dataset_dir: Path to category folder (e.g., datasets/mvtec_ad/bottle).

    Returns:
        image_paths: List of discovered image file paths.
        labels: Corresponding binary labels array (0 for normal, 1 for defect).
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths: List[str] = []
    labels: List[int] = []

    for root, _, files in os.walk(dataset_dir):
        # Determine if this directory contains normal or defective samples
        dir_name = os.path.basename(root).lower()
        is_normal = (dir_name == "good")

        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.join(root, f)
                image_paths.append(full_path)
                labels.append(0 if is_normal else 1)

    return image_paths, np.array(labels, dtype=int)


def extract_and_save_dataset_features(
    dataset_dir: str,
    output_dir: str = "datasets",
    batch_size: int = 32
) -> Dict[str, str]:
    """
    Complete end-to-end extraction job:
    1. Scans images in dataset_dir.
    2. Runs ResNet50 forward inference.
    3. Saves features.npy, labels.npy, and image_paths.json into output_dir.

    Args:
        dataset_dir: Root path of images to process.
        output_dir: Directory where features and metadata will be saved.
        batch_size: Batch size for inference.

    Returns:
        output_paths: Dictionary containing saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths, labels = scan_industrial_dataset(dataset_dir)

    if len(image_paths) == 0:
        raise ValueError(f"No valid image files found in {dataset_dir}")

    print(f"[EXTRACTOR] Found {len(image_paths)} images in {dataset_dir}.")
    print(f"[EXTRACTOR] Normal samples: {int(np.sum(labels == 0))}, Defect samples: {int(np.sum(labels == 1))}.")

    features = extract_features_from_image_paths(image_paths, batch_size=batch_size)

    features_path = os.path.join(output_dir, "features.npy")
    labels_path = os.path.join(output_dir, "labels.npy")
    paths_file = os.path.join(output_dir, "image_paths.json")

    np.save(features_path, features)
    np.save(labels_path, labels)
    with open(paths_file, "w") as f:
        json.dump(image_paths, f, indent=2)

    print(f"[EXTRACTOR] Successfully exported feature matrix to {features_path} (shape: {features.shape})")
    return {
        "features_path": features_path,
        "labels_path": labels_path,
        "image_paths_file": paths_file
    }

