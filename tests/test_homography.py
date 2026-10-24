from __future__ import annotations

import numpy as np

from panorama.homography import (
    estimate_homography_dlt,
    estimate_homography_ransac,
    project_points,
)


def test_normalized_dlt_recovers_exact_transform() -> None:
    source = np.array(
        [[0, 0], [500, 0], [500, 300], [0, 300], [250, 120], [100, 250]],
        dtype=np.float64,
    )
    expected = np.array(
        [[1.05, 0.04, 120.0], [-0.02, 1.08, 35.0], [0.00012, -0.00008, 1.0]],
        dtype=np.float64,
    )
    destination = project_points(source, expected)

    estimated = estimate_homography_dlt(source, destination)
    error = np.linalg.norm(project_points(source, estimated) - destination, axis=1)
    assert float(np.max(error)) < 1e-6


def test_custom_ransac_rejects_outliers() -> None:
    rng = np.random.default_rng(123)
    expected = np.array(
        [[0.98, -0.025, 180.0], [0.015, 1.02, -28.0], [0.00008, -0.00005, 1.0]],
        dtype=np.float64,
    )
    inlier_source = rng.uniform([0, 0], [1200, 700], size=(90, 2))
    inlier_destination = project_points(inlier_source, expected)
    inlier_destination += rng.normal(0, 0.35, size=inlier_destination.shape)

    outlier_source = rng.uniform([0, 0], [1200, 700], size=(30, 2))
    outlier_destination = rng.uniform([0, 0], [1400, 850], size=(30, 2))
    source = np.vstack((inlier_source, outlier_source))
    destination = np.vstack((inlier_destination, outlier_destination))

    result = estimate_homography_ransac(
        source,
        destination,
        threshold=2.0,
        confidence=0.999,
        max_iterations=4000,
        random_seed=9,
    )
    predicted = project_points(inlier_source, result.matrix)
    error = np.linalg.norm(predicted - project_points(inlier_source, expected), axis=1)

    assert result.inlier_count >= 85
    assert float(np.median(error)) < 0.25
    assert result.iterations <= 4000
