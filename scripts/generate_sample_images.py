"""
Sample Industrial Dataset Generator
Generates a lightweight mock MVTec AD 'bottle' visual dataset for immediate local testing.
Creates:
  datasets/mvtec_ad/bottle/train/good/ (15 normal bottles)
  datasets/mvtec_ad/bottle/test/good/ (5 normal bottles)
  datasets/mvtec_ad/bottle/test/broken_large/ (5 defective bottles with cracks)
  datasets/mvtec_ad/bottle/test/contamination/ (5 defective bottles with dirt/blotches)
"""

import os
import numpy as np
from PIL import Image, ImageDraw


def draw_bottle(broken=False, contaminated=False, variation_seed=0):
    np.random.seed(variation_seed)
    # 256x256 image with light gray industrial background
    img = Image.new("RGB", (256, 256), color=(235, 238, 240))
    draw = ImageDraw.Draw(img)

    # Slight color and position variation simulating conveyor belt
    offset_x = np.random.randint(-4, 5)
    offset_y = np.random.randint(-4, 5)

    base_color = (60, 140, 180)  # Blue glass bottle

    # Bottle body: (x0, y0, x1, y1)
    bx0, by0 = 85 + offset_x, 90 + offset_y
    bx1, by1 = 171 + offset_x, 230 + offset_y
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=15, fill=base_color, outline=(40, 100, 130), width=3)

    # Bottle neck
    nx0, ny0 = 108 + offset_x, 35 + offset_y
    nx1, ny1 = 148 + offset_x, 90 + offset_y
    draw.rectangle([nx0, ny0, nx1, ny1], fill=base_color, outline=(40, 100, 130), width=3)

    # Bottle cap
    cx0, cy0 = 104 + offset_x, 20 + offset_y
    cx1, cy1 = 152 + offset_x, 35 + offset_y
    draw.rectangle([cx0, cy0, cx1, cy1], fill=(220, 180, 50), outline=(160, 130, 30), width=2)

    # If Defect: Broken (Draw fracture crack)
    if broken:
        crack_points = [
            (bx0 + 20, by0 + 30),
            (bx0 + 45, by0 + 60),
            (bx0 + 30, by0 + 85),
            (bx0 + 65, by0 + 115)
        ]
        draw.line(crack_points, fill=(20, 20, 20), width=4)
        draw.polygon([(bx0 + 20, by0 + 30), (bx0 + 35, by0 + 40), (bx0 + 30, by0 + 55)], fill=(235, 238, 240))

    # If Defect: Contamination (Draw dark oil/dirt blotches)
    if contaminated:
        for _ in range(6):
            spot_x = np.random.randint(bx0 + 15, bx1 - 15)
            spot_y = np.random.randint(by0 + 20, by1 - 20)
            r = np.random.randint(6, 14)
            draw.ellipse([spot_x - r, spot_y - r, spot_x + r, spot_y + r], fill=(45, 30, 20))

    return img


def generate_sample_dataset(root_dir: str = "datasets/mvtec_ad/bottle"):
    subdirs = {
        "train/good": (15, False, False),
        "test/good": (5, False, False),
        "test/broken_large": (5, True, False),
        "test/contamination": (5, False, True),
    }

    print(f"[DATASET GENERATOR] Creating sample MVTec AD bottle images in {root_dir}...")
    total_created = 0

    for subpath, (count, broken, contaminated) in subdirs.items():
        folder = os.path.join(root_dir, subpath)
        os.makedirs(folder, exist_ok=True)

        for i in range(count):
            img = draw_bottle(broken=broken, contaminated=contaminated, variation_seed=total_created * 17)
            filename = f"img_{i:03d}.png"
            img.save(os.path.join(folder, filename))
            total_created += 1

    print(f"[DATASET GENERATOR] Successfully created {total_created} sample images:")
    print("  - 20 Normal images (15 in train/good, 5 in test/good)")
    print("  - 10 Defective images (5 broken_large, 5 contamination)")


if __name__ == "__main__":
    generate_sample_dataset()

