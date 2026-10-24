from __future__ import annotations

import cv2

from panorama import PanoramaStitcher, StitchConfig
from scripts.generate_sample_data import build_scene, create_crops


def test_end_to_end_manual_order() -> None:
    scene = build_scene(width=2100, height=560, seed=11)
    crops = create_crops(scene, count=3)
    stitcher = PanoramaStitcher(
        StitchConfig(
            feature_method="ORB",
            projection='planar', output_megapixels=None, auto_crop=False,
            ratio_threshold=0.82,
            min_matches=8,
            ransac_threshold=3.0,
            max_dimension=1200,
            exposure_compensation=True,
        )
    )
    result = stitcher.stitch(crops, ["left", "middle", "right"], manual_order=[0, 1, 2])

    assert result.panorama.shape[1] > 1800
    assert result.panorama.shape[0] >= 500
    assert len(result.pair_diagnostics) == 2
    assert all(item.inlier_count >= 6 for item in result.pair_diagnostics)


def test_end_to_end_automatic_order_with_shuffled_input() -> None:
    scene = build_scene(width=2300, height=600, seed=21)
    crops = create_crops(scene, count=4)
    shuffled = [crops[2], crops[0], crops[3], crops[1]]
    stitcher = PanoramaStitcher(
        StitchConfig(
            feature_method="ORB",
            projection='planar', output_megapixels=None, auto_crop=False,
            ratio_threshold=0.84,
            min_matches=8,
            ransac_threshold=3.5,
            max_dimension=1200,
        )
    )
    result = stitcher.stitch(
        shuffled,
        ["third", "first", "fourth", "second"],
        automatic_order=True,
    )

    assert result.ordering is not None
    assert result.panorama.shape[1] > 1950
    assert set(result.ordered_names) == {"first", "second", "third", "fourth"}
    assert result.ordered_names in (
        ["first", "second", "third", "fourth"],
        ["fourth", "third", "second", "first"],
    )
