"""Custom feather blending and lightweight exposure compensation."""

from __future__ import annotations

import numpy as np


def rectangular_feather_weights(height: int, width: int, power: float = 1.0) -> np.ndarray:
    """Build a centre-weighted map using distance to the closest image edge.

    This calculation is implemented directly with NumPy rather than using an
    OpenCV blender.  After the map is warped with its image, overlapping pixels
    are combined by normalized weighted averaging.
    """

    if height < 1 or width < 1 or power <= 0:
        raise ValueError("Invalid feather weight dimensions or power.")
    y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1))
    x = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1))
    distance = np.minimum(y[:, None], x[None, :]).astype(np.float32)
    distance /= max(float(np.max(distance)), 1.0)
    return np.maximum(distance, 1e-3) ** power


def _estimate_gain(reference: np.ndarray, candidate: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    if np.count_nonzero(overlap) < 500:
        return np.ones(3, dtype=np.float32)
    gains = np.ones(3, dtype=np.float32)
    for channel in range(3):
        ref_values = reference[..., channel][overlap]
        new_values = candidate[..., channel][overlap]
        usable = (ref_values > 8) & (ref_values < 247) & (new_values > 8) & (new_values < 247)
        if np.count_nonzero(usable) < 200:
            continue
        ratio = np.median(ref_values[usable] / np.maximum(new_values[usable], 1.0))
        gains[channel] = float(np.clip(ratio, 0.67, 1.5))
    return gains


def feather_blend(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    weights: list[np.ndarray],
    exposure_compensation: bool = True,
) -> np.ndarray:
    """Blend already-warped images into a uint8 BGR panorama."""

    if not images or not (len(images) == len(masks) == len(weights)):
        raise ValueError("Images, masks and weights must be non-empty and equal in length.")
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("All warped images must have the same shape.")

    numerator = np.zeros(shape, dtype=np.float32)
    denominator = np.zeros(shape[:2], dtype=np.float32)

    for image, mask, weight in zip(images, masks, weights, strict=True):
        valid = np.asarray(mask, dtype=bool)
        corrected = image.astype(np.float32)
        if exposure_compensation and np.any(denominator > 1e-6):
            current = numerator / np.maximum(denominator[..., None], 1e-6)
            overlap = valid & (denominator > 1e-3)
            corrected *= _estimate_gain(current, corrected, overlap)[None, None, :]
            corrected = np.clip(corrected, 0, 255)

        effective_weight = np.where(valid, np.asarray(weight, dtype=np.float32), 0.0)
        numerator += corrected * effective_weight[..., None]
        denominator += effective_weight

    output = numerator / np.maximum(denominator[..., None], 1e-6)
    output[denominator <= 1e-6] = 0
    return np.clip(output, 0, 255).astype(np.uint8)


def crop_to_content(image: np.ndarray, mask: np.ndarray, padding: int = 0) -> np.ndarray:
    """Crop away the unused black canvas according to the warped validity mask."""

    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return image
    top = max(0, int(rows.min()) - padding)
    bottom = min(image.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(image.shape[1], int(columns.max()) + padding + 1)
    return image[top:bottom, left:right]
