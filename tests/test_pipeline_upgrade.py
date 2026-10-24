import numpy as np
import pytest
from panorama import PanoramaStitcher, StitchConfig
from panorama.exceptions import PanoramaError
from scripts.generate_sample_data import build_scene, create_crops


def test_new_defaults_and_invalid_modes():
    config = StitchConfig()
    assert config.projection == 'auto'
    assert config.output_megapixels == 100
    assert config.auto_crop
    with pytest.raises(ValueError):
        PanoramaStitcher(StitchConfig(projection='wrong'))
    with pytest.raises(ValueError):
        PanoramaStitcher(StitchConfig(output_megapixels=float('nan')))


def test_auto_translation_pipeline_exports_requested_size_after_crop():
    images = create_crops(build_scene(width=1900, height=540, seed=13), count=3)
    result = PanoramaStitcher(StitchConfig(feature_method='ORB', ratio_threshold=.84,
                                          min_matches=8, output_megapixels=.3)).stitch(images)
    assert result.projection == 'planar'
    assert result.coverage_verified
    assert .3e6 <= np.prod(result.panorama.shape[:2]) < .302e6
    assert result.projection_reason
    assert result.quality['planar']['crop_fraction'] > .9


def test_cylindrical_translation_rejected_instead_of_fabricating_calibration():
    images = create_crops(build_scene(width=1600, height=540, seed=31), count=2)
    with pytest.raises(PanoramaError, match='focal|camera|Camera'):
        PanoramaStitcher(StitchConfig(projection='cylindrical', feature_method='ORB',
                                      ratio_threshold=.84, output_megapixels=.2)).stitch(images)


def test_product_core_does_not_delegate_algorithms():
    import ast
    from pathlib import Path
    forbidden = {'Stitcher_create', 'findHomography', 'BFMatcher', 'knnMatch',
                 'warpPerspective', 'remap', 'Rodrigues', 'distanceTransform',
                 'least_squares', 'minimize', 'GraphCutSeamFinder', 'FeatherBlender',
                 'MultiBandBlender'}
    for path in (Path(__file__).parents[1]/'panorama').glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not (calls & forbidden), (path.name, calls & forbidden)


def test_nonoverlap_and_invalid_images_fail_before_large_output():
    rng = np.random.default_rng(123)
    images = [rng.integers(0, 256, (220, 300, 3), dtype=np.uint8) for _ in range(2)]
    with pytest.raises(PanoramaError):
        PanoramaStitcher(StitchConfig(feature_method='ORB')).stitch(images, automatic_order=True)
    with pytest.raises(PanoramaError):
        PanoramaStitcher().stitch([np.zeros((80, 80, 3), dtype=np.uint8)]*2)
