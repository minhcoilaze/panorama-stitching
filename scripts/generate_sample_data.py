"""Generate a copyright-safe synthetic panorama dataset for local testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_scene(width: int = 2600, height: int = 720, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = np.zeros((height, width, 3), dtype=np.uint8)

    # Smooth sky/ground background.
    for row in range(height):
        fraction = row / height
        if fraction < 0.62:
            color = np.array([210, 170, 85]) * (1 - fraction) + np.array([235, 215, 170]) * fraction
        else:
            color = np.array([80, 125, 75]) * (1 - fraction) + np.array([50, 85, 45]) * fraction
        scene[row, :] = np.clip(color, 0, 255)

    # Buildings with repeated windows provide stable local features.
    building_colors = [(120, 90, 65), (80, 105, 145), (145, 120, 85), (95, 85, 125)]
    for building_index, x in enumerate(range(80, width - 100, 330)):
        building_width = int(rng.integers(230, 310))
        building_height = int(rng.integers(220, 410))
        top = height - 110 - building_height
        color = building_colors[building_index % len(building_colors)]
        cv2.rectangle(scene, (x, top), (x + building_width, height - 110), color, -1)
        cv2.rectangle(scene, (x, top), (x + building_width, height - 110), (30, 30, 30), 3)
        for window_y in range(top + 25, height - 140, 55):
            for window_x in range(x + 22, x + building_width - 25, 58):
                cv2.rectangle(scene, (window_x, window_y), (window_x + 28, window_y + 30), (230, 210, 120), -1)
                cv2.line(scene, (window_x + 14, window_y), (window_x + 14, window_y + 30), (80, 80, 80), 1)
        cv2.putText(
            scene,
            f"B{building_index + 1}",
            (x + 12, top - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    # Unique signs, trees, lines and texture prevent ambiguous repeated matches.
    for index in range(75):
        center = (int(rng.integers(20, width - 20)), int(rng.integers(30, height - 40)))
        radius = int(rng.integers(3, 14))
        color = tuple(int(value) for value in rng.integers(20, 245, size=3))
        cv2.circle(scene, center, radius, color, -1, cv2.LINE_AA)
    for index, x in enumerate(range(160, width, 420)):
        cv2.line(scene, (x, height - 155), (x, height - 55), (55, 40, 25), 8)
        cv2.circle(scene, (x, height - 180), 52, (35, 120 + (index % 2) * 30, 45), -1)
        cv2.putText(scene, str(index + 1), (x - 14, height - 170), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.line(scene, (0, height - 108), (width, height - 108), (245, 245, 245), 5)
    cv2.putText(scene, "SYNTHETIC PANORAMA - DIGITAL IMAGE PROCESSING", (420, 75), cv2.FONT_HERSHEY_DUPLEX, 1.35, (25, 35, 55), 3, cv2.LINE_AA)
    return scene


def create_crops(scene: np.ndarray, count: int = 4) -> list[np.ndarray]:
    crop_width = 1000
    step = (scene.shape[1] - crop_width) // (count - 1)
    crops = []
    brightness = [0.93, 1.0, 1.07, 0.97]
    for index in range(count):
        start = min(index * step, scene.shape[1] - crop_width)
        crop = scene[:, start : start + crop_width].copy()
        factor = brightness[index % len(brightness)]
        crop = np.clip(crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        crops.append(crop)
    return crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("sample_data"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scene = build_scene(seed=args.seed)
    cv2.imwrite(str(args.output / "reference_panorama.png"), scene)
    for index, crop in enumerate(create_crops(scene), start=1):
        cv2.imwrite(str(args.output / f"sample_{index:02d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Created a reference panorama and four overlapping crops in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
