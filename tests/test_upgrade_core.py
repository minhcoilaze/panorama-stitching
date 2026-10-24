from __future__ import annotations

import numpy as np
import pytest

from panorama.optimization import robust_least_squares
from panorama.features import nearest_two
from panorama.cropping import largest_valid_rectangle
from panorama.seams import minimum_cut


def test_robust_optimizer_recovers_line_despite_outlier():
    x = np.arange(20, dtype=float)
    y = 2 * x + 3
    y[-1] += 200
    result = robust_least_squares(lambda p: p[0] * x + p[1] - y, np.zeros(2), max_iterations=60)
    assert abs(result.parameters[0] - 2) < .08
    assert abs(result.parameters[1] - 3) < .4
    assert result.cost < result.initial_cost


@pytest.mark.parametrize('method', ['SIFT', 'ORB'])
def test_custom_nearest_neighbors(method):
    if method == 'SIFT':
        a = np.array([[1., 0.], [9., 0.]], dtype=np.float32)
        b = np.array([[0., 0.], [3., 0.], [10., 0.]], dtype=np.float32)
    else:
        a = np.array([[0], [255]], dtype=np.uint8)
        b = np.array([[1], [15], [254]], dtype=np.uint8)
    indices, distances = nearest_two(a, b, method)
    assert indices[:, 0].tolist() == [0, 2]
    assert np.allclose(distances[:, 0], 1)


def test_largest_rectangle_avoids_internal_holes_and_black_content():
    mask = np.ones((6, 9), dtype=bool)
    mask[:, :2] = False
    mask[2, 5] = False
    top, left, bottom, right = largest_valid_rectangle(mask)
    assert mask[top:bottom, left:right].all()
    # Independent exhaustive oracle for a small mask.
    best = max((b-t)*(r-l) for t in range(6) for b in range(t+1,7)
               for l in range(9) for r in range(l+1,10) if mask[t:b,l:r].all())
    assert (bottom-top)*(right-left) == best


def test_largest_rectangle_empty_mask_rejected():
    with pytest.raises(ValueError):
        largest_valid_rectangle(np.zeros((4, 5), dtype=bool))


def test_custom_min_cut_matches_known_bottleneck():
    # source 0 --3--> 1 --1--> 2 --3--> sink 3
    flow, reachable = minimum_cut(4, [(0, 1, 3.), (1, 2, 1.), (2, 3, 3.)], 0, 3)
    assert abs(flow-1) < 1e-6
    assert reachable.tolist() == [True, True, False, False]


def test_custom_min_cut_matches_exhaustive_oracle():
    from itertools import product
    rng = np.random.default_rng(8)
    for count in range(3, 7):
        for _ in range(6):
            edges = [(a,b,float(rng.integers(1,10))) for a in range(count) for b in range(count)
                     if a != b and rng.random() < .3]
            flow, reachable = minimum_cut(count, edges, 0, count-1)
            costs = []
            for bits in product([False, True], repeat=count-2):
                part = [True, *bits, False]
                costs.append(sum(c for a,b,c in edges if part[a] and not part[b]))
            assert flow == pytest.approx(min(costs))
            assert sum(c for a,b,c in edges if reachable[a] and not reachable[b]) == pytest.approx(flow)
