"""Homography estimation implemented with NumPy.

This module deliberately does not call ``cv2.findHomography``.  It implements
normalised Direct Linear Transform (DLT) and a robust RANSAC loop so the core
algorithm can be explained and assessed independently from OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, log

import numpy as np

from .exceptions import HomographyError


@dataclass(slots=True)
class HomographyResult:
    matrix: np.ndarray
    inlier_mask: np.ndarray
    errors: np.ndarray
    iterations: int

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def mean_inlier_error(self) -> float:
        if not np.any(self.inlier_mask):
            return float("inf")
        return float(np.mean(self.errors[self.inlier_mask]))


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise HomographyError(f"{name} must have shape (N, 2).")
    if len(array) < 4:
        raise HomographyError("At least four point correspondences are required.")
    if not np.all(np.isfinite(array)):
        raise HomographyError(f"{name} contains NaN or infinite values.")
    return array


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply Hartley normalisation so mean distance from origin is sqrt(2)."""

    points = _as_points(points, "points")
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance < np.finfo(np.float64).eps:
        raise HomographyError("Point configuration is degenerate.")

    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def estimate_homography_dlt(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Estimate a projective mapping from ``source`` to ``destination``."""

    source = _as_points(source, "source")
    destination = _as_points(destination, "destination")
    if len(source) != len(destination):
        raise HomographyError("Source and destination must have equal length.")

    src, source_transform = normalize_points(source)
    dst, destination_transform = normalize_points(destination)

    matrix_rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        matrix_rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        matrix_rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    design = np.asarray(matrix_rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 8:
        raise HomographyError("Correspondences do not define a unique homography.")

    # Four correspondences produce an 8x9 design matrix.  ``full_matrices``
    # must stay enabled so Vh includes the ninth (null-space) vector.
    # Only the minimal 8x9 system needs the full null-space basis. For N>=5,
    # reduced SVD avoids allocating the huge (2N)x(2N) left singular matrix.
    _, _, vh = np.linalg.svd(design, full_matrices=design.shape[0] < 9)
    normalized_h = vh[-1].reshape(3, 3)
    homography = np.linalg.inv(destination_transform) @ normalized_h @ source_transform

    denominator = homography[2, 2]
    if abs(denominator) < 1e-12:
        norm = np.linalg.norm(homography)
        if norm < 1e-12:
            raise HomographyError("Estimated homography is numerically unstable.")
        homography /= norm
    else:
        homography /= denominator

    if not np.all(np.isfinite(homography)) or abs(np.linalg.det(homography)) < 1e-12:
        raise HomographyError("Estimated homography is singular or unstable.")
    return homography


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Transform 2-D points by a 3x3 homography."""

    points = np.asarray(points, dtype=np.float64)
    homography = np.asarray(homography, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or homography.shape != (3, 3):
        raise HomographyError("Invalid point or homography shape.")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (homography @ homogeneous.T).T
    scale = projected[:, 2]
    valid = np.abs(scale) > 1e-12
    output = np.full((len(points), 2), np.inf, dtype=np.float64)
    output[valid] = projected[valid, :2] / scale[valid, None]
    return output


def reprojection_errors(
    source: np.ndarray, destination: np.ndarray, homography: np.ndarray
) -> np.ndarray:
    predicted = project_points(source, homography)
    return np.linalg.norm(predicted - np.asarray(destination, dtype=np.float64), axis=1)


def _general_position(points: np.ndarray) -> bool:
    """Reject four-point samples containing an almost-collinear triple."""

    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    minimum_area_twice = 1e-4 * span * span
    for a, b, c in combinations(points, 3):
        ab = b - a
        ac = c - a
        area_twice = abs(ab[0] * ac[1] - ab[1] * ac[0])
        if area_twice <= minimum_area_twice:
            return False
    return True


def estimate_homography_ransac(
    source: np.ndarray,
    destination: np.ndarray,
    threshold: float = 4.0,
    confidence: float = 0.995,
    max_iterations: int = 3000,
    random_seed: int = 42,
) -> HomographyResult:
    """Robustly estimate homography using a custom four-point RANSAC loop."""

    source = _as_points(source, "source")
    destination = _as_points(destination, "destination")
    if len(source) != len(destination):
        raise HomographyError("Source and destination must have equal length.")
    if threshold <= 0 or max_iterations < 1 or not 0 < confidence < 1:
        raise HomographyError("Invalid RANSAC parameters.")

    rng = np.random.default_rng(random_seed)
    sample_size = 4
    best_mask = np.zeros(len(source), dtype=bool)
    best_errors = np.full(len(source), np.inf, dtype=np.float64)
    best_count = 0
    best_median = float("inf")
    required_iterations = max_iterations
    completed = 0

    while completed < min(max_iterations, required_iterations):
        completed += 1
        indices = rng.choice(len(source), size=sample_size, replace=False)
        sample_source = source[indices]
        sample_destination = destination[indices]
        if not _general_position(sample_source) or not _general_position(sample_destination):
            continue

        try:
            candidate = estimate_homography_dlt(sample_source, sample_destination)
            errors = reprojection_errors(source, destination, candidate)
        except (HomographyError, np.linalg.LinAlgError, FloatingPointError):
            continue

        mask = errors <= threshold
        count = int(np.count_nonzero(mask))
        median = float(np.median(errors[mask])) if count else float("inf")
        if count > best_count or (count == best_count and median < best_median):
            best_mask = mask
            best_errors = errors
            best_count = count
            best_median = median

            inlier_ratio = count / len(source)
            success_probability = inlier_ratio**sample_size
            if success_probability >= 1.0:
                required_iterations = completed
            elif success_probability > 0:
                denominator = log(max(1e-12, 1.0 - success_probability))
                estimate = ceil(log(1.0 - confidence) / denominator)
                required_iterations = max(completed, min(required_iterations, estimate))

    if best_count < 4:
        raise HomographyError("RANSAC could not find four consistent correspondences.")

    # Refit on all consensus points, then update consensus once more.
    try:
        refined = estimate_homography_dlt(source[best_mask], destination[best_mask])
        refined_errors = reprojection_errors(source, destination, refined)
        refined_mask = refined_errors <= threshold
        if np.count_nonzero(refined_mask) >= 4:
            refined = estimate_homography_dlt(
                source[refined_mask], destination[refined_mask]
            )
            refined_errors = reprojection_errors(source, destination, refined)
            refined_mask = refined_errors <= threshold
    except (HomographyError, np.linalg.LinAlgError) as exc:
        raise HomographyError("RANSAC consensus could not be refined.") from exc

    return HomographyResult(refined, refined_mask, refined_errors, completed)
