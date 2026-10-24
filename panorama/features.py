"""Local feature detection, descriptor matching and visual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .exceptions import InsufficientFeaturesError, InsufficientMatchesError


@dataclass(slots=True)
class FeatureSet:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray
    method: str


@dataclass(slots=True)
class MatchSet:
    matches: list[cv2.DMatch]
    points_a: np.ndarray
    points_b: np.ndarray

    def __len__(self) -> int:
        return len(self.matches)


def _detector(method: str) -> tuple[cv2.Feature2D, int]:
    method = method.upper()
    if method == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise InsufficientFeaturesError(
                "SIFT is unavailable in this OpenCV build. Select ORB instead."
            )
        return cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03), cv2.NORM_L2
    if method == "ORB":
        return (
            cv2.ORB_create(
                nfeatures=6000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=19,
                fastThreshold=10,
            ),
            cv2.NORM_HAMMING,
        )
    raise ValueError("Feature method must be SIFT or ORB.")


def detect_features(image: np.ndarray, method: str = "SIFT") -> FeatureSet:
    if image is None or image.size == 0:
        raise InsufficientFeaturesError("The image is empty.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector, _ = _detector(method)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 4:
        raise InsufficientFeaturesError(
            "The image has too few features. Use a sharper image with more texture."
        )
    return FeatureSet(list(keypoints), descriptors, method.upper())


def nearest_two(a: np.ndarray, b: np.ndarray, method: str,
                block_size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Exact blocked KNN: squared Euclidean (SIFT) or XOR/popcount (ORB).

    Memory is bounded by a block of queries, never an N*M*descriptor tensor.
    """
    if len(b) < 2:
        raise InsufficientMatchesError('At least two candidate descriptors are needed.')
    indices = np.empty((len(a), 2), dtype=np.int32)
    distances = np.empty((len(a), 2), dtype=np.float32)
    if method.upper() == 'SIFT':
        reference = np.asarray(b, dtype=np.float32)
    else:
        # For 0/1 vectors, XOR/popcount equals |a|+|b|-2*a.b exactly.
        # Bit unpacking plus blocked matrix multiplication avoids a giant XOR
        # tensor; all integer sums <= descriptor bits are exact in float32.
        reference = np.unpackbits(b, axis=1).astype(np.float32)
    reference_norm = np.sum(reference*reference, axis=1)
    for start in range(0, len(a), block_size):
        query = a[start:start+block_size]
        query = (np.asarray(query, dtype=np.float32) if method.upper() == 'SIFT'
                 else np.unpackbits(query, axis=1).astype(np.float32))
        matrix = np.maximum(np.sum(query*query, axis=1)[:, None]+reference_norm[None, :]
                            - 2*(query @ reference.T), 0.)
        # Two linear scans avoid the copying/partitioning cost for k=2.
        rows = np.arange(len(query))
        first = np.argmin(matrix, axis=1)
        first_distance = matrix[rows, first].copy()
        matrix[rows, first] = np.inf
        second = np.argmin(matrix, axis=1)
        two = np.column_stack([first, second])
        values = np.column_stack([first_distance, matrix[rows, second]])
        indices[start:start+len(query)] = two
        distances[start:start+len(query)] = np.sqrt(values) if method.upper() == 'SIFT' else values
    return indices, distances


def match_features(
    features_a: FeatureSet,
    features_b: FeatureSet,
    ratio_threshold: float = 0.75,
    min_matches: int = 4,
    mutual_check: bool = True,
) -> MatchSet:
    if features_a.method != features_b.method:
        raise ValueError("Both feature sets must use the same detector.")
    forward_indices, forward_distances = nearest_two(
        features_a.descriptors, features_b.descriptors, features_a.method)
    keep = forward_distances[:, 0] < ratio_threshold*forward_distances[:, 1]

    if mutual_check:
        reverse_indices, reverse_distances = nearest_two(
            features_b.descriptors, features_a.descriptors, features_a.method)
        best = forward_indices[:, 0]
        keep &= ((reverse_indices[best, 0] == np.arange(len(keep))) &
                 (reverse_distances[best, 0] < ratio_threshold*reverse_distances[best, 1]))

    forward = [cv2.DMatch(int(i), int(forward_indices[i, 0]), float(forward_distances[i, 0]))
               for i in np.flatnonzero(keep)]

    forward.sort(key=lambda match: match.distance)
    if len(forward) < min_matches:
        raise InsufficientMatchesError(
            f"Only {len(forward)} reliable matches were found; at least {min_matches} are required."
        )

    points_a = np.float64([features_a.keypoints[m.queryIdx].pt for m in forward])
    points_b = np.float64([features_b.keypoints[m.trainIdx].pt for m in forward])
    return MatchSet(forward, points_a, points_b)


def draw_keypoints(image: np.ndarray, features: FeatureSet) -> np.ndarray:
    return cv2.drawKeypoints(
        image,
        features.keypoints,
        None,
        color=(0, 220, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_matches(
    image_a: np.ndarray,
    features_a: FeatureSet,
    image_b: np.ndarray,
    features_b: FeatureSet,
    matches: MatchSet,
    inlier_mask: np.ndarray | None = None,
    limit: int = 80,
) -> np.ndarray:
    selected = matches.matches[:limit]
    mask = None
    if inlier_mask is not None:
        mask = [int(value) for value in np.asarray(inlier_mask)[: len(selected)]]
    return cv2.drawMatches(
        image_a,
        features_a.keypoints,
        image_b,
        features_b.keypoints,
        selected,
        None,
        matchColor=(34, 197, 94),
        singlePointColor=(239, 68, 68),
        matchesMask=mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
