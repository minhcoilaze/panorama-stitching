"""Explicit planar/cylindrical mapping and explainable projection selection."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .camera import CameraModel, intrinsic
from .exceptions import HomographyError
from .homography import project_points


@dataclass
class Projection:
    kind: str
    shapes: list[tuple[int, int]]
    transforms: list[np.ndarray] | None = None
    cameras: CameraModel | None = None
    fit_error: float = 0.
    max_fit_error: float = 8.
    _centers: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.kind not in ('planar', 'cylindrical'):
            raise ValueError('Unknown projection.')
        if self.kind == 'cylindrical':
            if self.cameras is None:
                raise ValueError('Cylindrical projection requires calibrated cameras.')
            axes = [r[:, 2] for r in self.cameras.rotations]
            self._centers = np.unwrap([np.arctan2(a[0], a[2]) for a in axes])
        else:
            if self.transforms is None:
                raise ValueError('Planar projection requires transforms.')
            self._centers = np.zeros(len(self.shapes))

    @property
    def focal_scale(self):
        return float(self.cameras.focals[self.cameras.reference]) if self.cameras else 1.

    def project(self, index: int, points: np.ndarray) -> np.ndarray:
        if self.kind == 'planar':
            return project_points(points, self.transforms[index])
        k = intrinsic(self.shapes[index], self.cameras.focals[index])
        rays = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(k).T
        world = rays @ self.cameras.rotations[index].T
        angle = np.arctan2(world[:, 0], world[:, 2])
        center = self._centers[index]
        angle = center+(angle-center+np.pi)%(2*np.pi)-np.pi
        vertical = world[:, 1]/np.maximum(np.hypot(world[:, 0], world[:, 2]), 1e-12)
        return self.focal_scale*np.column_stack([angle, vertical])

    def inverse(self, index: int, x: np.ndarray, y: np.ndarray, world_rays=None):
        if self.kind == 'planar':
            inverse = np.linalg.inv(self.transforms[index])
            z = inverse[2, 0]*x+inverse[2, 1]*y+inverse[2, 2]
            nonzero = np.abs(z) > 1e-10
            safe_z = np.where(nonzero, z, 1.)
            u = (inverse[0, 0]*x+inverse[0, 1]*y+inverse[0, 2])/safe_z
            v = (inverse[1, 0]*x+inverse[1, 1]*y+inverse[1, 2])/safe_z
        else:
            angle, height = x/self.focal_scale, y/self.focal_scale
            r = self.cameras.rotations[index]
            sin, cos = ((np.sin(angle), np.cos(angle)) if world_rays is None else world_rays)
            cx = r[0, 0]*sin+r[1, 0]*height+r[2, 0]*cos
            cy = r[0, 1]*sin+r[1, 1]*height+r[2, 1]*cos
            z = r[0, 2]*sin+r[1, 2]*height+r[2, 2]*cos
            nonzero = z > 1e-10
            safe_z = np.where(nonzero, z, 1.)
            h, w = self.shapes[index]
            f = self.cameras.focals[index]
            u, v = f*cx/safe_z+(w-1)/2, f*cy/safe_z+(h-1)/2
            nonzero &= np.abs(angle-self._centers[index]) < np.pi
        h, w = self.shapes[index]
        valid = nonzero & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u <= w-1) & (v <= h-1)
        return u, v, valid

    def bounds(self) -> tuple[float, float, float, float]:
        boundaries = []
        for i, (h, w) in enumerate(self.shapes):
            a = np.linspace(0, 1, 129)
            points = np.vstack([np.column_stack([a*(w-1), np.zeros_like(a)]),
                                np.column_stack([a*(w-1), np.full_like(a, h-1)]),
                                np.column_stack([np.zeros_like(a), a*(h-1)]),
                                np.column_stack([np.full_like(a, w-1), a*(h-1)])])
            if self.kind == 'planar':
                denominator = np.column_stack([points, np.ones(len(points))]) @ self.transforms[i][2]
                if np.min(denominator)*np.max(denominator) <= 0:
                    raise HomographyError('The planar transform crosses a projection horizon; choose cylindrical mode.')
            boundaries.append(self.project(i, points))
        points = np.vstack(boundaries)
        if not np.isfinite(points).all():
            raise HomographyError('Projection bounds are not finite.')
        minimum, maximum = points.min(0), points.max(0)
        if np.max(maximum-minimum) > 1e7:
            raise HomographyError('Projection geometry is unstable; output size cannot fix this distortion.')
        return float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])

    def metrics(self) -> dict[str, float]:
        error = self.cameras.mean_error if self.cameras else self.fit_error
        if not np.isfinite(error) or error > self.max_fit_error:
            raise HomographyError('Projection alignment error is too large to accept this model.')
        bounds = self.bounds()  # Reject horizon crossings before taking Jacobian SVDs.
        scales, anisotropy = [], []
        for i, (h, w) in enumerate(self.shapes):
            xx, yy = np.meshgrid(np.linspace(2, w-3, 9), np.linspace(2, h-3, 7))
            points = np.column_stack([xx.ravel(), yy.ravel()])
            p = self.project(i, points)
            dx = self.project(i, points+[.5, 0])-p
            dy = self.project(i, points+[0, .5])-p
            jac = np.stack([dx, dy], axis=2)*2
            singular = np.linalg.svd(jac, compute_uv=False)
            scales.extend(np.sqrt(np.prod(singular, axis=1)))
            anisotropy.extend(singular[:, 0]/np.maximum(singular[:, 1], 1e-8))
        scales = np.asarray(scales)
        low, high = np.percentile(scales, [5, 95])
        return {'stretch': float(high/max(low, 1e-8)),
                'anisotropy': float(np.percentile(anisotropy, 95)),
                'canvas_area': float((bounds[2]-bounds[0])*(bounds[3]-bounds[1])),
                'error': self.cameras.mean_error if self.cameras else self.fit_error}


def choose_projection(planar: Projection | None, cylindrical: Projection | None,
                      draft_metrics: dict | None = None):
    metrics, failures = {}, []
    for item in (planar, cylindrical):
        if item is None:
            continue
        try:
            metrics[item.kind] = item.metrics()
            if draft_metrics and item.kind in draft_metrics:
                metrics[item.kind].update(draft_metrics[item.kind])
        except HomographyError as error:
            failures.append(str(error))
    if not metrics:
        raise HomographyError('No reliable projection is available. '+' '.join(failures))
    if 'planar' not in metrics:
        return cylindrical, 'Chọn hình trụ: phép chiếu phẳng không ổn định.', metrics
    if 'cylindrical' not in metrics:
        return planar, 'Chọn phẳng: mô hình camera quay chưa đủ tin cậy hoặc cảnh chủ yếu là tịnh tiến.', metrics
    p, c = metrics['planar'], metrics['cylindrical']
    if c['error'] > max(4., 2.5*p['error']):
        return planar, 'Chọn phẳng: sai số mô hình camera quay cao hơn mức chấp nhận.', metrics
    # Geometry-based rule, independent of output MP / machine memory.
    p_score = p['stretch']*np.sqrt(p['anisotropy'])
    c_score = c['stretch']*np.sqrt(c['anisotropy'])
    if ((p['stretch'] > 2.2 or p['anisotropy'] > 2.) and c_score < .8*p_score
            and c.get('crop_fraction', 1.) >= .5*p.get('crop_fraction', 1.)):
        return cylindrical, (f"Chọn hình trụ: mức giãn phẳng {p['stretch']:.2f}×, "
                              f"hình trụ {c['stretch']:.2f}×; sai số camera {c['error']:.2f} px."), metrics
    return planar, 'Chọn phẳng: biến dạng thấp; giữ đường thẳng và cấu trúc cảnh.', metrics
