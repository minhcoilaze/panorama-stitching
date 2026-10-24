"""Custom-core panorama pipeline: registration, camera models and tiled export."""
from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from .camera import Correspondence, estimate_cameras, refine_planar, validate_alignment
from .config import StitchConfig
from .exceptions import HomographyError, InsufficientMatchesError, InvalidImageError
from .features import FeatureSet, MatchSet, detect_features, draw_keypoints, draw_matches, match_features
from .geometry import make_thumbnail, resize_to_limit
from .homography import HomographyResult, estimate_homography_ransac, reprojection_errors
from .ordering import OrderingResult, order_images_automatically
from .projection import Projection, choose_projection
from .rendering import build_draft, render_panorama, prepare_seams

ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class PairDiagnostic:
    source_position: int
    destination_position: int
    match_count: int
    inlier_count: int
    mean_error: float
    visualization: np.ndarray


@dataclass(slots=True)
class StitchResult:
    panorama: np.ndarray
    ordered_indices: list[int]
    ordered_names: list[str]
    elapsed_seconds: float
    pair_diagnostics: list[PairDiagnostic] = field(default_factory=list)
    keypoint_visualizations: list[np.ndarray] = field(default_factory=list)
    warped_previews: list[np.ndarray] = field(default_factory=list)
    ordering: OrderingResult | None = None
    resized_shapes: list[tuple[int, int]] = field(default_factory=list)
    projection: str = 'planar'
    projection_reason: str = ''
    quality: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    coverage_verified: bool = False
    interpolated: bool = False
    natural_shape: tuple[int, int] = (0, 0)
    crop_fraction: float = 1.
    requested_megapixels: float | None = None
    camera_focals: list[float] = field(default_factory=list)


class PanoramaStitcher:
    def __init__(self, config: StitchConfig | None = None) -> None:
        self.config = config or StitchConfig()
        self.config.validate()

    @staticmethod
    def _validate_images(images: list[np.ndarray], names: list[str]) -> None:
        if not 2 <= len(images) <= 10:
            raise InvalidImageError('Select between 2 and 10 images.')
        if len(images) != len(names):
            raise InvalidImageError('The image and filename counts do not match.')
        for index, image in enumerate(images):
            if not isinstance(image, np.ndarray) or not image.size:
                raise InvalidImageError(f'Image {index+1} could not be decoded.')
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise InvalidImageError(f'Image {index+1} must be a uint8 three-channel color image.')
            if min(image.shape[:2]) < 64:
                raise InvalidImageError(f'Image {index+1} is too small for reliable matching.')

    @threadpool_limits.wrap(limits=4, user_api='blas')
    def stitch(self, images: list[np.ndarray], names: list[str] | None = None,
               automatic_order: bool = False, manual_order: list[int] | None = None,
               progress: ProgressCallback | None = None) -> StitchResult:
        started = perf_counter()
        if names is None:
            names = [f'Image {index+1}' for index in range(len(images))]
        self._validate_images(images, names)

        def update(value, message):
            if progress:
                progress(float(np.clip(value, 0, 1)), message)

        update(.02, 'Chuẩn bị ảnh đăng ký; giữ ảnh gốc để xuất')
        resized_scales = [resize_to_limit(image, self.config.max_dimension) for image in images]
        resized, scales = [x[0] for x in resized_scales], [x[1] for x in resized_scales]
        features: list[FeatureSet] = []
        for i, image in enumerate(resized):
            features.append(detect_features(image, self.config.feature_method))
            update(.04+.12*(i+1)/len(images), f'Đặc trưng: {i+1}/{len(images)}')
        cache = {}
        ordering = None
        if automatic_order:
            ordering = order_images_automatically(features, self.config,
                        lambda v, m: update(.16+.16*v, m), pair_cache=cache)
            order = ordering.indices
        elif manual_order is not None:
            if sorted(manual_order) != list(range(len(images))):
                raise InvalidImageError('Manual order must contain every image exactly once.')
            order = list(manual_order)
        else:
            order = list(range(len(images)))
        original_images = [images[i] for i in order]
        resized, features = [resized[i] for i in order], [features[i] for i in order]
        scales = [scales[i] for i in order]
        names_ordered = [names[i] for i in order]
        count, reference = len(images), len(images)//2
        shapes = [image.shape[:2] for image in resized]
        transforms = [None]*count
        transforms[reference] = np.eye(3)
        diagnostics, correspondences = [], []
        directed = [(a, a+1) for a in range(reference-1, -1, -1)]
        directed += [(a, a-1) for a in range(reference+1, count)]

        for number, (source, destination) in enumerate(directed):
            old_a, old_b = order[source], order[destination]
            if (old_a, old_b) in cache:
                matched, robust = cache[old_a, old_b]
            elif (old_b, old_a) in cache:
                previous, previous_robust = cache[old_b, old_a]
                matched = MatchSet([cv2.DMatch(m.trainIdx, m.queryIdx, m.distance)
                                    for m in previous.matches], previous.points_b, previous.points_a)
                h = np.linalg.inv(previous_robust.matrix)
                errors = reprojection_errors(matched.points_a, matched.points_b, h)
                # Re-evaluate reverse-direction pixel errors; do not reuse a
                # forward mask when local scales differ.
                robust = HomographyResult(h, errors <= self.config.ransac_threshold,
                                          errors, previous_robust.iterations)
            else:
                matched = match_features(features[source], features[destination],
                            self.config.ratio_threshold, self.config.min_matches)
                robust = estimate_homography_ransac(matched.points_a, matched.points_b,
                            threshold=self.config.ransac_threshold, confidence=self.config.ransac_confidence,
                            max_iterations=self.config.ransac_max_iterations,
                            random_seed=self.config.random_seed+source*count+destination)
            minimum = max(6, self.config.min_matches//2)
            if len(matched) < self.config.min_matches or robust.inlier_count < minimum or robust.inlier_count/len(matched) < .25:
                raise InsufficientMatchesError(f'Images {source+1} and {destination+1} do not have enough reliable overlap.')
            points_a, points_b = matched.points_a[robust.inlier_mask], matched.points_b[robust.inlier_mask]
            for points, shape in ((points_a, shapes[source]), (points_b, shapes[destination])):
                coverage = np.ptp(points, axis=0)/np.array([shape[1], shape[0]])
                if np.prod(coverage) < .008:
                    raise InsufficientMatchesError('Matches are concentrated in too small an area for stable geometry.')
            correspondences.append(Correspondence(source, destination, points_a, points_b, robust.matrix))
            transforms[source] = transforms[destination] @ robust.matrix
            visual = draw_matches(resized[source], features[source], resized[destination],
                                  features[destination], matched, robust.inlier_mask)
            diagnostics.append(PairDiagnostic(source, destination, len(matched), robust.inlier_count,
                                              robust.mean_inlier_error, make_thumbnail(visual, 1200)))
            update(.32+.10*(number+1)/(count-1), f'DLT/RANSAC: cặp {number+1}/{count-1}')

        # Additional non-adjacent graph edges are used only when strongly
        # supported, avoiding weak repeated foliage matches in the global fit.
        position = {old: new for new, old in enumerate(order)}
        adjacent = {frozenset((p.source, p.destination)) for p in correspondences}
        for (old_a, old_b), (matched, robust) in cache.items():
            a, b = position[old_a], position[old_b]
            if frozenset((a, b)) in adjacent or robust.inlier_count < 30 or robust.inlier_count/len(matched) < .5:
                continue
            correspondences.append(Correspondence(a, b, matched.points_a[robust.inlier_mask],
                                                  matched.points_b[robust.inlier_mask], robust.matrix))

        update(.44, 'Tinh chỉnh hình học toàn bộ bộ ảnh')
        candidates, drafts, failures = {}, {}, {}
        planar = None
        max_fit_error = max(4., self.config.ransac_threshold*2)
        try:
            transforms = refine_planar(shapes, correspondences, transforms, reference)
            planar_error = float(np.mean(validate_alignment(correspondences, transforms, max_fit_error)))
            planar = Projection('planar', shapes, transforms=transforms,
                                fit_error=planar_error, max_fit_error=max_fit_error)
            candidates['planar'] = planar
        except (HomographyError, np.linalg.LinAlgError) as error:
            failures['planar'] = str(error)
            if self.config.projection == 'planar':
                raise HomographyError(str(error)) from error
        if self.config.projection != 'planar':
            try:
                cameras = estimate_cameras(shapes, correspondences, reference,
                                            max_error=max(4., self.config.ransac_threshold*2))
                candidates['cylindrical'] = Projection('cylindrical', shapes, cameras=cameras,
                                                     max_fit_error=max_fit_error)
            except HomographyError as error:
                failures['cylindrical'] = str(error)
                if self.config.projection == 'cylindrical':
                    raise
        requested = ([self.config.projection] if self.config.projection != 'auto' else list(candidates))
        for kind in requested:
            update(.50, f'Đánh giá phép chiếu {kind} và vùng crop')
            try:
                draft = build_draft(candidates[kind], resized, exposure=self.config.exposure_compensation,
                                    max_dimension=self.config.draft_dimension, seam_dimension=self.config.seam_dimension,
                                    compute_seams=False)
                draft.rectangle(self.config.auto_crop)
                drafts[kind] = draft
            except HomographyError as error:
                failures[kind] = str(error)
                if self.config.projection != 'auto':
                    raise
        if self.config.projection == 'auto':
            metrics = {kind: {'crop_fraction': draft.crop_fraction} for kind, draft in drafts.items()}
            selected, reason, quality = choose_projection(
                planar if 'planar' in drafts else None,
                candidates.get('cylindrical') if 'cylindrical' in drafts else None, metrics)
        else:
            selected = candidates[self.config.projection]
            reason = 'Sử dụng phép chiếu '+('phẳng' if selected.kind == 'planar' else 'hình trụ')+' theo lựa chọn của bạn.'
            quality = {selected.kind: {**selected.metrics(), 'crop_fraction': drafts[selected.kind].crop_fraction}}
        quality['unavailable'] = failures
        draft = drafts[selected.kind]
        update(.56, 'Tìm đường nối graph-cut tự cài đặt')
        prepare_seams(draft, self.config.seam_dimension)
        warnings = []
        if selected.kind == 'planar' and quality['planar']['stretch'] > 2.2:
            warnings.append('Phép chiếu phẳng kéo giãn mạnh ở hai đầu; thử Hình trụ hoặc Tự động để so sánh.')
        update(.58, 'Đã chọn phép chiếu; dựng ảnh gốc theo từng vùng')
        rendered = render_panorama(draft, original_images, scales,
                      target_megapixels=self.config.output_megapixels, auto_crop=self.config.auto_crop,
                      tile_rows=self.config.tile_rows, tile_columns=self.config.tile_columns,
                      progress=lambda v, m: update(.58+.40*v, m))
        if rendered.interpolated:
            warnings.append('Có nội suy để đạt độ phân giải đã chọn; số pixel tăng không tạo thêm chi tiết thật.')
        if selected.cameras and selected.cameras.mean_error > 3:
            warnings.append('Còn sai số căn chỉnh; kiểm tra vùng gần camera và vật chuyển động.')
        update(1., 'Hoàn thành panorama')
        return StitchResult(
            panorama=rendered.image, ordered_indices=order, ordered_names=names_ordered,
            elapsed_seconds=perf_counter()-started, pair_diagnostics=diagnostics,
            keypoint_visualizations=[make_thumbnail(draw_keypoints(im, f), 700) for im, f in zip(resized, features)],
            warped_previews=[make_thumbnail(np.rint(im).astype(np.uint8), 700) for im in draft.images],
            ordering=ordering, resized_shapes=shapes, projection=selected.kind,
            projection_reason=reason, quality=quality, warnings=warnings,
            coverage_verified=rendered.coverage_verified, interpolated=rendered.interpolated,
            natural_shape=rendered.natural_shape, crop_fraction=draft.crop_fraction,
            requested_megapixels=self.config.output_megapixels,
            camera_focals=selected.cameras.focals.tolist() if selected.cameras else [])
