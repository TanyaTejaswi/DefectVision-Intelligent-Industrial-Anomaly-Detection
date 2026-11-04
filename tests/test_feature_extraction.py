"""
Unit tests for Feature Extraction module (src/feature_extraction.py)
"""

import os
import numpy as np
import pytest
from PIL import Image
import torch
from src.feature_extraction import (
    get_image_transform,
    build_resnet50_extractor,
    extract_batch_features,
    scan_industrial_dataset
)


def test_get_image_transform():
    transform = get_image_transform(image_size=224)
    # Create a random RGB test image
    dummy_img = Image.new("RGB", (300, 400), color=(128, 64, 32))
    tensor = transform(dummy_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_build_resnet50_extractor_and_batch_inference():
    # Use pretrained=False for fast offline unit testing
    device = torch.device("cpu")
    model = build_resnet50_extractor(device=device, pretrained=False)

    assert isinstance(model.fc, torch.nn.Identity)

    # Test forward pass with a batch of 2 dummy images
    dummy_batch = torch.randn(2, 3, 224, 224)
    embeddings = extract_batch_features(dummy_batch, model, device)

    assert embeddings.shape == (2, 2048)
    assert isinstance(embeddings, np.ndarray)
    assert not np.isnan(embeddings).any()


def test_scan_industrial_dataset(tmp_path):
    # Create mock MVTec AD folder structure
    dataset_dir = tmp_path / "mock_mvtec"
    good_dir = dataset_dir / "train" / "good"
    defect_dir = dataset_dir / "test" / "crack"

    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(defect_dir, exist_ok=True)

    # Save 2 good images and 1 defective image
    img = Image.new("RGB", (100, 100), color="white")
    img.save(good_dir / "normal_01.png")
    img.save(good_dir / "normal_02.png")
    img.save(defect_dir / "defect_01.png")

    image_paths, labels = scan_industrial_dataset(str(dataset_dir))

    assert len(image_paths) == 3
    assert len(labels) == 3
    # Normal images should be labeled 0, defective labeled 1
    assert np.sum(labels == 0) == 2
    assert np.sum(labels == 1) == 1

