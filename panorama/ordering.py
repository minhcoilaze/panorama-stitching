"""Automatic image ordering based on pairwise overlap quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import StitchConfig
from .exceptions import HomographyError, InsufficientMatchesError, OrderingError
from .features import FeatureSet, match_features
from .homography import estimate_homography_ransac


@dataclass(slots=True)
class OrderingResult:
    indices: list[int]
    score_matrix: np.ndarray
    path_score: float


def _maximum_weight_path(scores: np.ndarray) -> tuple[list[int], float]:
    """Find the best Hamiltonian path with Held-Karp dynamic programming."""

    count = len(scores)
    # (visited_mask, last_node) -> (score, path_tuple)
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
        (1 << node, node): (0.0, (node,)) for node in range(count)
    }
    for size in range(1, count):
        next_states = dict(states)
        for (mask, last), (value, path) in states.items():
            if len(path) != size:
                continue
            for candidate in range(count):
                if mask & (1 << candidate):
                    continue
                new_mask = mask | (1 << candidate)
                new_value = value + float(scores[last, candidate])
                key = (new_mask, candidate)
                previous = next_states.get(key)
                if previous is None or new_value > previous[0]:
                    next_states[key] = (new_value, path + (candidate,))
        states = next_states

    full_mask = (1 << count) - 1
    candidates = [value for (mask, _), value in states.items() if mask == full_mask]
    if not candidates:
        raise OrderingError("Could not build an ordering path.")
    best_value, best_path = max(candidates, key=lambda item: item[0])
    return list(best_path), float(best_value)


def order_images_automatically(
    features: list[FeatureSet],
    config: StitchConfig,
    progress: Callable[[float, str], None] | None = None,
    pair_cache: dict | None = None,
) -> OrderingResult:
    """Order images by maximizing reliable adjacent overlap.

    Pair scores are RANSAC inlier counts.  The final left/right orientation is
    inferred from the median horizontal position of corresponding features.
    """

    count = len(features)
    if count < 2:
        raise OrderingError("At least two images are required for ordering.")
    scores = np.zeros((count, count), dtype=np.float64)
    directions = np.zeros((count, count), dtype=np.float64)
    total_pairs = count * (count - 1) // 2
    completed_pairs = 0

    for first in range(count):
        for second in range(first + 1, count):
            completed_pairs += 1
            if progress:
                progress(
                    completed_pairs / max(total_pairs, 1),
                    f"Checking overlap {first + 1}/{second + 1}",
                )
            try:
                matched = match_features(
                    features[first],
                    features[second],
                    ratio_threshold=config.ratio_threshold,
                    min_matches=4,
                )
                robust = estimate_homography_ransac(
                    matched.points_a,
                    matched.points_b,
                    threshold=config.ransac_threshold,
                    confidence=min(config.ransac_confidence, 0.99),
                    max_iterations=min(config.ransac_max_iterations, 1000),
                    random_seed=config.random_seed + first * count + second,
                )
            except (InsufficientMatchesError, HomographyError):
                continue

            inliers = robust.inlier_count
            inlier_ratio = inliers / max(len(matched), 1)
            if inliers < 4 or inlier_ratio < 0.2:
                continue
            if pair_cache is not None:
                pair_cache[first, second] = (matched, robust)
            scores[first, second] = scores[second, first] = float(inliers)

            inlier_points_a = matched.points_a[robust.inlier_mask]
            inlier_points_b = matched.points_b[robust.inlier_mask]
            horizontal_delta = float(
                np.median(inlier_points_a[:, 0] - inlier_points_b[:, 0])
            )
            directions[first, second] = horizontal_delta
            directions[second, first] = -horizontal_delta

    path, path_score = _maximum_weight_path(scores)
    minimum_edge_score = max(4, min(config.min_matches, 8))
    weak_edges = [
        (left, right, scores[left, right])
        for left, right in zip(path, path[1:])
        if scores[left, right] < minimum_edge_score
    ]
    if weak_edges:
        raise OrderingError(
            "Automatic ordering is uncertain because some images do not overlap. "
            "Use manual ordering or choose images with at least 30% overlap."
        )

    orientation_vote = sum(
        np.sign(directions[left, right]) * scores[left, right]
        for left, right in zip(path, path[1:])
    )
    if orientation_vote < 0:
        path.reverse()

    return OrderingResult(path, scores, path_score)
