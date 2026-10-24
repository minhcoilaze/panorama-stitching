from __future__ import annotations
import numpy as np
import pytest
from panorama.camera import Correspondence, estimate_cameras, rotation_matrix
from panorama.projection import Projection, choose_projection
from panorama.homography import project_points
from panorama.exceptions import HomographyError


def rotating_fixture():
    shape = (600, 1000)
    k = np.array([[800., 0, 499.5], [0, 800., 299.5], [0, 0, 1]])
    rotations = [rotation_matrix(np.array([0., a, 0.])) for a in [-.65, 0., .65]]
    rng = np.random.default_rng(9)
    pairs = []
    for a, b in [(0, 1), (1, 2)]:
        h = k @ rotations[b].T @ rotations[a] @ np.linalg.inv(k)
        p = rng.uniform([5, 5], [994, 594], size=(2000, 2))
        q = project_points(p, h)
        keep = (q[:, 0] > 5) & (q[:, 0] < 994) & (q[:, 1] > 5) & (q[:, 1] < 594)
        pairs.append(Correspondence(a, b, p[keep][:100], q[keep][:100], h))
    return [shape]*3, pairs


def test_rotations_are_orthogonal():
    r = rotation_matrix(np.array([.2, -.5, .1]))
    assert np.allclose(r.T@r, np.eye(3), atol=1e-10)
    assert np.linalg.det(r) == pytest.approx(1.)


def test_camera_recovers_known_focal_and_rotations():
    shapes, pairs = rotating_fixture()
    camera = estimate_cameras(shapes, pairs, reference=1)
    assert np.allclose(camera.focals, 800, rtol=.03)
    assert camera.mean_error < .1
    projection = Projection('cylindrical', shapes, cameras=camera)
    for pair in pairs:
        a = projection.project(pair.source, pair.points_source)
        b = projection.project(pair.destination, pair.points_destination)
        assert np.max(np.linalg.norm(a-b, axis=1)) < .2


def test_cylindrical_inverse_round_trip_and_automatic_choice():
    shapes, pairs = rotating_fixture()
    camera = estimate_cameras(shapes, pairs, reference=1)
    cylindrical = Projection('cylindrical', shapes, cameras=camera)
    planar = Projection('planar', shapes, transforms=[pairs[0].homography, np.eye(3),
                                                    np.linalg.inv(pairs[1].homography)])
    points = np.array([[20., 30.], [500., 300.], [980., 580.]])
    for i in range(3):
        projected = cylindrical.project(i, points)
        x, y, valid = cylindrical.inverse(i, projected[:, 0], projected[:, 1])
        assert valid.all()
        assert np.allclose(np.column_stack([x, y]), points, atol=1e-6)
    selected, reason, metrics = choose_projection(planar, cylindrical)
    assert selected.kind == 'cylindrical'
    assert reason and metrics['planar']['stretch'] > metrics['cylindrical']['stretch']


def test_translation_prefers_planar_and_does_not_fake_camera_calibration():
    shapes = [(300, 500)]*2
    h = np.array([[1., 0, 200], [0, 1, 0], [0, 0, 1]])
    p = np.array([[20, 20], [100, 20], [20, 200], [100, 200]], dtype=float)
    pair = Correspondence(0, 1, p, project_points(p, h), h)
    with pytest.raises(HomographyError):
        estimate_cameras(shapes, [pair], reference=1)
    planar = Projection('planar', shapes, transforms=[h, np.eye(3)])
    selected, _, _ = choose_projection(planar, None)
    assert selected.kind == 'planar'


def test_auto_does_not_accept_bad_planar_fit_even_when_distortion_is_small():
    planar = Projection('planar', [(300, 500)]*2, transforms=[np.eye(3), np.eye(3)], fit_error=30.)
    with pytest.raises(HomographyError):
        choose_projection(planar, None)


def test_inconsistent_pair_cycle_is_rejected_after_global_refinement():
    from panorama.camera import refine_planar, validate_alignment
    shapes = [(300, 500)]*3
    p = np.array([[40., 30.], [180., 30.], [40., 180.], [180., 180.], [100., 100.]])
    pairs = []
    for a, b, shift in [(0, 1, 30), (1, 2, 30), (0, 2, 100)]:
        h = np.array([[1., 0, shift], [0, 1, 0], [0, 0, 1]])
        pairs.append(Correspondence(a, b, p, project_points(p, h), h))
    initial = [pairs[0].homography, np.eye(3), np.linalg.inv(pairs[1].homography)]
    refined = refine_planar(shapes, pairs, initial, 1)
    with pytest.raises(HomographyError):
        validate_alignment(pairs, refined, max_error=8.)
