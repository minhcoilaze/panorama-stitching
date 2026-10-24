from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import pytest
from panorama.projection import Projection
from panorama.rendering import bilinear_sample, build_draft, render_panorama


def simple_scene():
    yy, xx = np.indices((80, 200))
    scene = np.stack([xx, yy*2, (xx+yy)%200], axis=2).astype(np.uint8)
    images = [scene[:, :130].copy(), scene[:, 70:].copy()]
    h = np.array([[1., 0, 70], [0, 1, 0], [0, 0, 1]])
    projection = Projection('planar', [im.shape[:2] for im in images], transforms=[np.eye(3), h])
    return scene, images, projection


def test_custom_bilinear_interpolation_and_validity():
    image = np.array([[0, 10], [20, 30]], dtype=np.float32)
    values, valid = bilinear_sample(image, np.array([.5, -1]), np.array([.5, 0]))
    assert values[0] == pytest.approx(15.)
    assert valid.tolist() == [True, False]


def test_tiled_render_matches_source_and_is_independent_of_tile_size():
    scene, images, projection = simple_scene()
    draft = build_draft(projection, images, exposure=False)
    first = render_panorama(draft, images, [1., 1.], target_megapixels=None,
                            auto_crop=False, tile_rows=13, tile_columns=31)
    second = render_panorama(draft, images, [1., 1.], target_megapixels=None,
                             auto_crop=False, tile_rows=37, tile_columns=93)
    assert first.image.shape == scene.shape
    assert np.array_equal(first.image, second.image)
    assert np.max(np.abs(first.image.astype(float)-scene)) <= 1


def test_requested_megapixels_after_closed_crop_and_temporary_cleanup(tmp_path):
    _, images, projection = simple_scene()
    draft = build_draft(projection, images, exposure=False)
    result = render_panorama(draft, images, [1., 1.], target_megapixels=.12,
                             auto_crop=True, memmap_threshold_mp=0, temp_directory=tmp_path)
    assert .12e6 <= np.prod(result.image.shape[:2]) < .121e6
    assert result.coverage_verified
    assert result.interpolated
    assert isinstance(result.image, np.memmap)
    filename = Path(result.image.filename)
    assert filename.exists()
    del result
    gc.collect()
    assert not filename.exists()


def test_black_images_remain_valid_after_crop():
    _, images, projection = simple_scene()
    images = [np.zeros_like(im) for im in images]
    draft = build_draft(projection, images, exposure=False)
    result = render_panorama(draft, images, [1., 1.], target_megapixels=None, auto_crop=True)
    assert result.coverage_verified
    assert result.image.shape[0] > 60
    assert not result.image.any()


def test_original_sampling_uses_actual_per_axis_resize_scales():
    # cv2.resize rounds axes separately: 201x47 -> 100x23, not scalar-exact.
    yy, xx = np.indices((47, 201))
    original = np.stack([xx, yy*4, xx*0], axis=2).astype(np.uint8)
    projection = Projection('planar', [(23, 100)], transforms=[np.eye(3)])
    draft_image = np.zeros((23, 100, 3), dtype=np.uint8)
    draft = build_draft(projection, [draft_image], exposure=False)
    result = render_panorama(draft, [original], [100/201], target_megapixels=None, auto_crop=False)
    work_y = np.linspace(0, 22, result.image.shape[0])
    source_y = (work_y+.5)/(23/47)-.5
    expected = np.rint(source_y*4).astype(np.uint8)
    assert np.array_equal(result.image[:, 10, 1], expected)


@pytest.mark.parametrize('kind', ['planar', 'cylindrical'])
def test_compiled_custom_kernel_agrees_with_numpy_reference(kind):
    from panorama.camera import CameraModel, rotation_matrix
    from panorama.kernels import ACCELERATED
    assert ACCELERATED
    _, images, projection = simple_scene()
    if kind == 'cylindrical':
        camera = CameraModel(np.array([120., 120.]),
            [rotation_matrix(np.array([.03, a, .01])) for a in [-.2, .2]], 1, 0., 0., 0)
        projection = Projection(kind, [im.shape[:2] for im in images], cameras=camera)
    draft = build_draft(projection, images, exposure=False)
    plain = render_panorama(draft, images, [1., 1.], target_megapixels=.07, accelerate=False)
    fast = render_panorama(draft, images, [1., 1.], target_megapixels=.07, accelerate=True)
    assert np.max(np.abs(plain.image.astype(float)-fast.image)) <= 1
    assert fast.coverage_verified
