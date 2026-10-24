"""Rotation-camera autocalibration and joint refinement, without OpenCV solvers."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .exceptions import HomographyError
from .homography import project_points
from .optimization import robust_least_squares


@dataclass(slots=True)
class Correspondence:
    source: int
    destination: int
    points_source: np.ndarray
    points_destination: np.ndarray
    homography: np.ndarray


@dataclass(slots=True)
class CameraModel:
    focals: np.ndarray
    rotations: list[np.ndarray]  # camera-to-world, column-vector convention
    reference: int
    mean_error: float
    initial_error: float
    iterations: int


def rotation_matrix(vector: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map, implemented explicitly."""
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    x, y, z = vector
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    if angle < 1e-8:
        return np.eye(3)+skew+.5*(skew@skew)
    return np.eye(3)+(np.sin(angle)/angle)*skew+((1-np.cos(angle))/angle**2)*(skew@skew)


def intrinsic(shape: tuple[int, int], focal: float) -> np.ndarray:
    h, w = shape
    return np.array([[focal, 0., (w-1)/2], [0., focal, (h-1)/2], [0., 0., 1.]])


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    if np.linalg.det(matrix) < 0:
        matrix = -matrix
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1., 1., np.linalg.det(u@vt)]) @ vt


def _sample_pairs(pairs: list[Correspondence], limit: int = 140) -> list[Correspondence]:
    result = []
    for pair in pairs:
        indices = np.linspace(0, len(pair.points_source)-1,
                              min(limit, len(pair.points_source)), dtype=int)
        result.append(Correspondence(pair.source, pair.destination, pair.points_source[indices],
                                     pair.points_destination[indices], pair.homography))
    return result


def _initial_focal(shapes, pairs):
    sizes = np.array([max(shape) for shape in shapes], dtype=float)
    projective = [np.linalg.norm(p.homography[2, :2])*sizes[p.source] for p in pairs]
    if max(projective, default=0) < .015:
        raise HomographyError('Camera focal length is not identifiable from near-translation matches; use planar mode.')

    def objective(log_ratio):
        focals = sizes*np.exp(log_ratio)
        errors = []
        for pair in pairs:
            a, b = pair.source, pair.destination
            mapping = np.linalg.inv(intrinsic(shapes[b], focals[b])) @ pair.homography @ intrinsic(shapes[a], focals[a])
            singular = np.linalg.svd(mapping, compute_uv=False)
            logarithms = np.log(np.maximum(singular, 1e-12))
            errors.append(float(np.sum((logarithms-logarithms.mean())**2)))
        return float(np.median(errors)+.25*np.mean(errors))

    grid = np.linspace(np.log(.15), np.log(8.), 64)
    scores = np.array([objective(value) for value in grid])
    best = int(np.argmin(scores))
    if best in (0, len(grid)-1):
        raise HomographyError('Camera focal estimation is unstable; the sequence may not be a rotating-camera panorama.')
    left, right = grid[best-1], grid[best+1]
    golden = (np.sqrt(5)-1)/2
    for _ in range(35):
        a, b = right-golden*(right-left), left+golden*(right-left)
        if objective(a) < objective(b):
            right = b
        else:
            left = a
    return float((left+right)/2), sizes


def estimate_cameras(shapes: list[tuple[int, int]], pairs: list[Correspondence],
                     reference: int, max_error: float = 8.) -> CameraModel:
    """Shared normalized focal for one fixed-lens sweep; H=Kj Rj.T Ri inv(Ki).

    A single focal/size ratio avoids fitting fictitious camera zoom to parallax.
    Cameras may have different registration resolutions. The reference rotation
    is fixed to remove the gauge freedom, with small rotation-vector updates.
    """
    if not pairs:
        raise HomographyError('No camera correspondences are available.')
    log_ratio, sizes = _initial_focal(shapes, pairs)
    focals = sizes*np.exp(log_ratio)
    ks = [intrinsic(shape, f) for shape, f in zip(shapes, focals)]
    rotations: list[np.ndarray | None] = [None]*len(shapes)
    rotations[reference] = np.eye(3)
    for _ in shapes:
        for pair in pairs:
            a, b = pair.source, pair.destination
            relative = _nearest_rotation(np.linalg.inv(ks[b]) @ pair.homography @ ks[a])
            if rotations[b] is not None and rotations[a] is None:
                rotations[a] = rotations[b] @ relative
            elif rotations[a] is not None and rotations[b] is None:
                rotations[b] = rotations[a] @ relative.T
    if any(r is None for r in rotations):
        raise HomographyError('The camera overlap graph is disconnected.')
    initial = [np.asarray(r) for r in rotations]
    variables = [i for i in range(len(shapes)) if i != reference]
    sampled = _sample_pairs(pairs)

    def unpack(parameters):
        focal = sizes*np.exp(np.clip(parameters[0], np.log(.10), np.log(12.)))
        rs = list(initial)
        for n, i in enumerate(variables):
            rs[i] = rotation_matrix(parameters[1+3*n:4+3*n]) @ initial[i]
        return focal, rs

    def residual(parameters, regularize=True):
        focal, rs = unpack(parameters)
        matrices = [intrinsic(shape, f) for shape, f in zip(shapes, focal)]
        errors = []
        for pair in sampled:
            a, b = pair.source, pair.destination
            h = matrices[b] @ rs[b].T @ rs[a] @ np.linalg.inv(matrices[a])
            predicted = project_points(pair.points_source, h)
            errors.append(np.nan_to_num(predicted-pair.points_destination, nan=1e5,
                                        posinf=1e5, neginf=-1e5).ravel())
        if regularize:
            # Weak focal prior regularizes ill-conditioned but usable sweeps.
            errors.append(np.array([.1*(parameters[0]-log_ratio)]))
        return np.concatenate(errors)

    parameters = np.zeros(1+3*len(variables))
    parameters[0] = log_ratio
    initial_error = float(np.mean(np.linalg.norm(residual(parameters, False).reshape(-1, 2), axis=1)))
    fit = robust_least_squares(residual, parameters, max_iterations=35, huber_delta=2.)
    focals, rotations = unpack(fit.parameters)
    mean_error = float(np.mean(np.linalg.norm(residual(fit.parameters, False).reshape(-1, 2), axis=1)))
    if not np.isfinite(mean_error) or mean_error > max_error:
        raise HomographyError(f'Rotation camera fit is unreliable ({mean_error:.2f} px); translation/parallax or mismatches may be present.')
    camera_transforms = [ks_ref @ r @ np.linalg.inv(intrinsic(shape, f))
                         for r, shape, f in zip(rotations, shapes, focals)
                         for ks_ref in [intrinsic(shapes[reference], focals[reference])]]
    validate_alignment(pairs, camera_transforms, max_error=max_error)
    if fit.parameters[0] <= np.log(.10) or fit.parameters[0] >= np.log(12.):
        raise HomographyError('Refined focal length reached an unstable boundary.')
    # Brown--Lowe up-vector: smallest eigenvector of camera horizontal axes.
    covariance = sum(np.outer(r[:, 0], r[:, 0]) for r in rotations)
    _, vectors = np.linalg.eigh(covariance)
    up = vectors[:, 0]
    if up @ rotations[reference][:, 1] < 0:
        up = -up
    target = np.array([0., 1., 0.])
    axis = np.cross(up, target)
    sine = np.linalg.norm(axis)
    if sine > 1e-8:
        correction = rotation_matrix(axis/sine*np.arctan2(sine, up@target))
        rotations = [correction@r for r in rotations]
    return CameraModel(focals, rotations, reference, mean_error, initial_error, fit.iterations)


def validate_alignment(pairs: list[Correspondence], transforms: list[np.ndarray],
                       max_error: float = 8.) -> list[float]:
    """Validate EACH graph edge after refinement, not just a pooled mean.

    A weak/mistaken cycle can be hidden by thousands of good matches elsewhere.
    Evaluate both directions in registration pixels so local scale cannot hide
    an unreliable edge. No candidate can pass on low distortion alone.
    """
    edge_errors = []
    for pair in pairs:
        try:
            h = np.linalg.inv(transforms[pair.destination])@transforms[pair.source]
            forward = np.linalg.norm(project_points(pair.points_source, h)-pair.points_destination, axis=1)
            backward = np.linalg.norm(project_points(pair.points_destination, np.linalg.inv(h))-pair.points_source, axis=1)
        except np.linalg.LinAlgError as error:
            raise HomographyError('Refined alignment is singular.') from error
        errors = np.maximum(forward, backward)
        mean = float(np.mean(errors))
        if (not np.isfinite(errors).all() or mean > max_error or
                np.mean(errors <= max_error) < .55):
            raise HomographyError(f'Refined alignment is inconsistent for images {pair.source+1}/{pair.destination+1} '
                                  f'({mean:.2f} px). Repeated patterns, parallax or conflicting matches may be present.')
        edge_errors.append(mean)
    return edge_errors


def refine_planar(shapes: list[tuple[int, int]], pairs: list[Correspondence],
                   transforms: list[np.ndarray], reference: int) -> list[np.ndarray]:
    """Joint symmetric transfer fit in normalized coordinates, reference fixed."""
    scales = [np.diag([float(max(shape)), float(max(shape)), 1.]) for shape in shapes]
    normalized = [np.linalg.inv(scales[reference])@h@scale for h, scale in zip(transforms, scales)]
    normalized = [h/h[2, 2] for h in normalized]
    indices = [i for i in range(len(shapes)) if i != reference]
    initial = np.concatenate([normalized[i].ravel()[:8] for i in indices])
    sampled = _sample_pairs(pairs, 100)

    def unpack(parameters):
        matrices = list(normalized)
        for j, i in enumerate(indices):
            matrices[i] = np.r_[parameters[8*j:8*j+8], 1.].reshape(3, 3)
        return [scales[reference]@h@np.linalg.inv(scale) for h, scale in zip(matrices, scales)]

    def residual(parameters):
        matrices = unpack(parameters)
        errors = []
        try:
            inverses = [np.linalg.inv(h) for h in matrices]
            for pair in sampled:
                a, b = pair.source, pair.destination
                forward = project_points(pair.points_source, inverses[b]@matrices[a])-pair.points_destination
                backward = project_points(pair.points_destination, inverses[a]@matrices[b])-pair.points_source
                errors.extend((forward.ravel(), backward.ravel()))
        except np.linalg.LinAlgError:
            return np.full(sum(4*len(p.points_source) for p in sampled), 1e8)
        return np.nan_to_num(np.concatenate(errors), nan=1e8, posinf=1e8, neginf=-1e8)

    fit = robust_least_squares(residual, initial, max_iterations=12, huber_delta=2.)
    return unpack(fit.parameters)
