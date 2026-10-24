"""Image resizing, projective canvas calculation and warping helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .exceptions import HomographyError
from .homography import project_points


@dataclass(slots=True)
class Canvas:
    width: int
    height: int
    translation: np.ndarray
    transforms: list[np.ndarray]


def resize_to_limit(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return image.copy(), 1.0
    scale = max_dimension / largest
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def image_corners(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )


def calculate_canvas(images: list[np.ndarray], transforms: list[np.ndarray],
                     max_pixels: int = 150_000_000) -> Canvas:
    transformed_corners = [
        project_points(image_corners(image), transform)
        for image, transform in zip(images, transforms, strict=True)
    ]
    all_corners = np.vstack(transformed_corners)
    if not np.all(np.isfinite(all_corners)):
        raise HomographyError("A transform projected the image outside a finite canvas.")

    minimum = np.floor(np.min(all_corners, axis=0))
    maximum = np.ceil(np.max(all_corners, axis=0))
    width = int(maximum[0] - minimum[0] + 1)
    height = int(maximum[1] - minimum[1] + 1)
    if width <= 0 or height <= 0:
        raise HomographyError("The panorama canvas has invalid dimensions.")
    if width > 100000 or height > 100000 or width * height > max_pixels:
        raise HomographyError(
            f"The estimated canvas is {width}x{height}, above the explicit {max_pixels:,}-pixel limit. "
            "Check projection distortion before allocating memory."
        )

    translation = np.array(
        [[1.0, 0.0, -minimum[0]], [0.0, 1.0, -minimum[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translated = [translation @ transform for transform in transforms]
    return Canvas(width, height, translation, translated)


def make_thumbnail(image: np.ndarray, max_width: int = 900) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image.copy()
    scale = max_width / image.shape[1]
    return cv2.resize(
        image,
        (max_width, max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
