from __future__ import annotations

import numpy as np

from panorama.blending import crop_to_content, feather_blend, rectangular_feather_weights


def test_feather_weights_are_largest_near_center() -> None:
    weights = rectangular_feather_weights(101, 151)
    assert weights.shape == (101, 151)
    assert weights[50, 75] > weights[0, 0]
    assert np.all(weights > 0)


def test_feather_blend_combines_overlap_smoothly() -> None:
    first = np.zeros((40, 100, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    first[:, :70] = (20, 40, 200)
    second[:, 30:] = (200, 40, 20)
    first_mask = np.zeros((40, 100), dtype=bool)
    second_mask = np.zeros_like(first_mask)
    first_mask[:, :70] = True
    second_mask[:, 30:] = True
    first_weight = np.tile(np.linspace(1.0, 0.05, 100), (40, 1)).astype(np.float32)
    second_weight = np.tile(np.linspace(0.05, 1.0, 100), (40, 1)).astype(np.float32)

    output = feather_blend(
        [first, second],
        [first_mask, second_mask],
        [first_weight, second_weight],
        exposure_compensation=False,
    )
    assert output.shape == first.shape
    assert output[20, 5, 2] > output[20, 5, 0]
    assert output[20, 95, 0] > output[20, 95, 2]
    assert 20 < int(output[20, 50, 0]) < 200
    assert 20 < int(output[20, 50, 2]) < 200


def test_crop_to_content_uses_mask_not_black_pixels() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=bool)
    mask[10:70, 15:100] = True
    cropped = crop_to_content(image, mask)
    assert cropped.shape == (60, 85, 3)
