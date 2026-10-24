"""Draft seams/crop plus bounded-memory, original-image inverse rendering.

No warp/remap/interpolation/blender implementation is delegated to OpenCV.
The only large persistent allocation is the final uint8 image (disk-backed
above the threshold); all floating point maps/accumulators live in tiles.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable
import weakref
import numpy as np
from .blending import _estimate_gain
from .cropping import erode_valid_mask, largest_valid_rectangle
from .exceptions import HomographyError, PanoramaError
from .projection import Projection
from .seams import graph_cut_ownership, feather_ownership
from .kernels import ACCELERATED, accumulate_tile


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Gather four neighbours and interpolate; return values and coverage."""
    h, w = image.shape[:2]
    x, y = np.broadcast_arrays(x, y)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1)
    u = np.clip(np.where(np.isfinite(x), x, 0), 0, w-1)
    v = np.clip(np.where(np.isfinite(y), y, 0), 0, h-1)
    x0, y0 = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    dx, dy = (u-x0).astype(np.float32), (v-y0).astype(np.float32)
    if image.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]
    top = image[y0, x0].astype(np.float32)*(1-dx)+image[y0, x1].astype(np.float32)*dx
    bottom = image[y1, x0].astype(np.float32)*(1-dx)+image[y1, x1].astype(np.float32)*dx
    return top*(1-dy)+bottom*dy, valid


@dataclass(slots=True)
class Draft:
    projection: Projection
    origin: tuple[float, float]
    step: float
    images: list[np.ndarray]
    masks: list[np.ndarray]
    coverage: np.ndarray
    gains: np.ndarray
    seam_weights: list[np.ndarray]
    crop_fraction: float = 1.

    @property
    def shape(self):
        return self.coverage.shape

    def rectangle(self, auto_crop: bool):
        h, w = self.shape
        if auto_crop:
            safe = erode_valid_mask(self.coverage, 2)
            try:
                top, left, bottom, right = largest_valid_rectangle(safe)
            except ValueError as error:
                raise HomographyError('No closed rectangular panorama remains after cropping.') from error
        else:
            top, left, bottom, right = 0, 0, h, w
        self.crop_fraction = ((bottom-top)*(right-left)/max(int(self.coverage.sum()), 1)
                              if auto_crop else 1.)
        return (self.origin[0]+left*self.step, self.origin[1]+top*self.step,
                self.origin[0]+(right-1)*self.step, self.origin[1]+(bottom-1)*self.step)


def build_draft(projection: Projection, images: list[np.ndarray], exposure: bool = True,
                max_dimension: int = 1000, seam_dimension: int = 160,
                compute_seams: bool = True) -> Draft:
    bounds = projection.bounds()
    span_x, span_y = bounds[2]-bounds[0], bounds[3]-bounds[1]
    step = max(1., max(span_x, span_y)/max(max_dimension-1, 1))
    width, height = int(np.ceil(span_x/step))+1, int(np.ceil(span_y/step))+1
    x, y = np.meshgrid(bounds[0]+np.arange(width)*step, bounds[1]+np.arange(height)*step)
    warped, masks = [], []
    for index, image in enumerate(images):
        u, v, geometry_valid = projection.inverse(index, x, y)
        values, valid = bilinear_sample(image, u, v)
        valid &= geometry_valid
        values[~valid] = 0
        warped.append(values)
        masks.append(valid)
    count = len(images)
    reference = count//2
    order = sorted(range(count), key=lambda i: (abs(i-reference), i))
    gains = np.ones((count, 3), dtype=np.float32)
    composite = np.zeros_like(warped[0])
    accumulated = np.zeros((height, width), dtype=np.float32)
    for i in order:
        mask = masks[i]
        if exposure and accumulated.any():
            current = composite/np.maximum(accumulated[..., None], 1.)
            gains[i] = _estimate_gain(current, warped[i], mask & (accumulated > 0))
        warped[i] = np.clip(warped[i]*gains[i], 0, 255)
        composite += warped[i]*mask[..., None]
        accumulated += mask
    draft = Draft(projection, (bounds[0], bounds[1]), step, warped, masks,
                  accumulated > 0, gains, [])
    if compute_seams:
        prepare_seams(draft, seam_dimension)
    return draft


def prepare_seams(draft: Draft, seam_dimension: int = 160):
    """Prepare only the chosen candidate's graph, not both draft candidates."""
    height, width = draft.shape
    warped, masks = draft.images, draft.masks
    count = len(masks)
    reference = count//2
    order = sorted(range(count), key=lambda i: (abs(i-reference), i))
    # Graph-cut runs on a small seam graph, never on the 100 MP output.
    factor = min(1., seam_dimension/max(width, height))
    sh, sw = max(2, round(height*factor)), max(2, round(width*factor))
    sx, sy = np.meshgrid(np.linspace(0, width-1, sw), np.linspace(0, height-1, sh))
    small_images = [bilinear_sample(im, sx, sy)[0] for im in warped]
    small_masks = [mask[np.rint(sy).astype(int), np.rint(sx).astype(int)] for mask in masks]
    labels = graph_cut_ownership(small_images, small_masks, order)
    draft.seam_weights = feather_ownership(labels, count, radius=3)


@dataclass(slots=True)
class RenderResult:
    image: np.ndarray
    coverage_verified: bool
    interpolated: bool
    crop_bounds: tuple[float, float, float, float]
    natural_shape: tuple[int, int]
    tile_count: int


def _release_mapping(mapping, filename):
    try:
        mapping.close()
    finally:
        try:
            os.unlink(filename)
        except OSError:
            pass


def _allocate_output(shape, threshold_mp, directory):
    pixels = shape[0]*shape[1]
    if pixels < threshold_mp*1e6:
        return np.empty(shape, dtype=np.uint8), None
    parent = Path(directory) if directory else Path(tempfile.gettempdir())
    if shutil.disk_usage(parent).free < pixels*3*1.2:
        raise PanoramaError('Not enough temporary disk space for the requested high-resolution output. No downscaling was applied.')
    descriptor, filename = tempfile.mkstemp(prefix='panorama-', suffix='.dat', dir=parent)
    os.close(descriptor)
    try:
        image = np.memmap(filename, mode='w+', dtype=np.uint8, shape=shape)
    except Exception:
        os.unlink(filename)
        raise
    cleanup = weakref.finalize(image, _release_mapping, image._mmap, filename)
    return image, cleanup


def render_panorama(draft: Draft, images: list[np.ndarray], scales: list[float],
                    target_megapixels: float | None = 100., auto_crop: bool = True,
                    tile_rows: int = 96, tile_columns: int = 1024,
                    memmap_threshold_mp: float = 24., temp_directory=None,
                    progress: Callable[[float, str], None] | None = None,
                    accelerate: bool = True) -> RenderResult:
    """Render selected MP AFTER crop, using original images, with no silent cap.

    Precise coverage is checked in every output tile. If a small interior hole
    escaped the draft grid, feed that location back into the draft mask and
    compute a new all-valid rectangle, then render again at the requested MP.
    """
    if target_megapixels is not None and (not np.isfinite(target_megapixels) or target_megapixels <= 0):
        raise ValueError('Output megapixels must be positive or None for native resolution.')
    if len(images) != len(scales) or len(images) != len(draft.masks):
        raise ValueError('Image, scale and projection counts must match.')
    if tile_rows < 1 or tile_columns < 1:
        raise ValueError('Tile dimensions must be positive.')
    for attempt in range(3):
        left, top, right, bottom = draft.rectangle(auto_crop)
        span_x, span_y = right-left, bottom-top
        if min(span_x, span_y) <= 0:
            raise HomographyError('The selected crop is too small.')
        # Natural density is based on the finest original input, not the
        # registration thumbnail. For equally scaled inputs this is exact.
        # resize rounds height and width independently. Derive the exact
        # pixel-centre transform from actual array shapes, not a nominal scalar.
        scale_x = [shape[1]/image.shape[1] for shape, image in zip(draft.projection.shapes, images)]
        scale_y = [shape[0]/image.shape[0] for shape, image in zip(draft.projection.shapes, images)]
        natural = (max(2, round(span_y/min(scale_y))+1), max(2, round(span_x/min(scale_x))+1))
        if target_megapixels is None:
            height, width = natural
        else:
            pixels = target_megapixels*1e6
            width = max(2, round(np.sqrt(pixels*span_x/span_y)))
            height = max(2, int(np.ceil(pixels/width)))
        if max(height, width) > 100_000 or height*width > 500_000_000:
            raise PanoramaError('The requested output dimensions exceed the supported export limits. Choose a smaller explicit output size.')
        try:
            output, cleanup = _allocate_output((height, width, 3), memmap_threshold_mp, temp_directory)
        except (MemoryError, OSError) as error:
            raise PanoramaError('Unable to allocate the requested output. Check RAM and temporary disk space; no downscaling was applied.') from error
        x_step, y_step = span_x/(width-1), span_y/(height-1)
        bad = False
        tiles = 0
        total_tiles = int(np.ceil(height/tile_rows))*int(np.ceil(width/tile_columns))
        try:
            for y_start in range(0, height, tile_rows):
                y_stop = min(height, y_start+tile_rows)
                for x_start in range(0, width, tile_columns):
                    x_stop = min(width, x_start+tile_columns)
                    x, y = np.meshgrid(left+np.arange(x_start, x_stop)*x_step,
                                       top+np.arange(y_start, y_stop)*y_step)
                    numerator = np.zeros((*x.shape, 3), dtype=np.float32)
                    denominator = np.zeros(x.shape, dtype=np.float32)
                    covered = np.zeros(x.shape, dtype=bool)
                    wx = (x-draft.origin[0])/(draft.step*max(draft.shape[1]-1, 1))*(draft.seam_weights[0].shape[1]-1)
                    wy = (y-draft.origin[1])/(draft.step*max(draft.shape[0]-1, 1))*(draft.seam_weights[0].shape[0]-1)
                    rays = ((np.sin(x/draft.projection.focal_scale), np.cos(x/draft.projection.focal_scale))
                            if draft.projection.kind == 'cylindrical' else None)
                    if accelerate and ACCELERATED:
                        fallback = np.zeros_like(numerator)
                        for i, image in enumerate(images):
                            projection = draft.projection
                            matrix = (np.linalg.inv(projection.transforms[i]) if projection.kind == 'planar'
                                      else projection.cameras.rotations[i])
                            focal = projection.cameras.focals[i] if projection.cameras else 1.
                            params = np.array([*projection.shapes[i], focal, projection.focal_scale,
                                               projection._centers[i], scale_x[i], scale_y[i]], dtype=float)
                            sine, cosine = rays if rays is not None else (x, x)
                            accumulate_tile(image, matrix, int(projection.kind == 'cylindrical'), params,
                                            x[0], y[:, 0], sine[0], cosine[0], draft.seam_weights[i],
                                            wx[0], wy[:, 0], draft.gains[i], numerator, denominator,
                                            fallback, covered)
                        values = numerator/np.maximum(denominator[..., None], 1e-8)
                        no_weight = denominator <= 1e-8
                        values[no_weight] = fallback[no_weight]
                    else:
                        for i, image in enumerate(images):
                            u, v, geometry_valid = draft.projection.inverse(i, x, y, rays)
                            # Convert registration pixel centres to original centres.
                            u, v = (u+.5)/scale_x[i]-.5, (v+.5)/scale_y[i]-.5
                            valid = (geometry_valid & (u >= 0) & (u <= image.shape[1]-1)
                                     & (v >= 0) & (v <= image.shape[0]-1))
                            covered |= valid
                            if not valid.any():
                                continue
                            weights = draft.seam_weights[i]
                            alpha, _ = bilinear_sample(weights, wx, wy)
                            active = valid & (alpha > 1e-8)
                            if not active.any():
                                continue
                            # Sample only pixels owned by this seam, instead of
                            # interpolating every overlapping image on every tile.
                            samples, _ = bilinear_sample(image, u[active], v[active])
                            samples = np.clip(samples*draft.gains[i], 0, 255)
                            numerator[active] += samples*alpha[active, None]
                            denominator[active] += alpha[active]
                        values = numerator/np.maximum(denominator[..., None], 1e-8)
                        missing = (denominator <= 1e-8) & covered
                        if missing.any():
                            # Rare exact-coverage pixels missed by the coarse seam
                            # grid still receive original content, never black fill.
                            for i, image in enumerate(images):
                                yy, xx = np.nonzero(missing)
                                if not len(yy):
                                    break
                                u, v, valid = draft.projection.inverse(i, x[yy, xx], y[yy, xx])
                                samples, sample_valid = bilinear_sample(image, (u+.5)/scale_x[i]-.5,
                                                                       (v+.5)/scale_y[i]-.5)
                                valid &= sample_valid
                                values[yy[valid], xx[valid]] = np.clip(samples[valid]*draft.gains[i], 0, 255)
                                missing[yy[valid], xx[valid]] = False
                    output[y_start:y_stop, x_start:x_stop] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
                    if auto_crop and not covered.all():
                        bad = True
                        # Invalidate the coarse neighbours so the next crop is
                        # conservative at sub-grid coverage boundaries too.
                        xx = np.rint((x[~covered]-draft.origin[0])/draft.step).astype(int)
                        yy = np.rint((y[~covered]-draft.origin[1])/draft.step).astype(int)
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                draft.coverage[np.clip(yy+dy, 0, draft.shape[0]-1),
                                               np.clip(xx+dx, 0, draft.shape[1]-1)] = False
                    tiles += 1
                    if progress and (tiles % 16 == 0 or tiles == total_tiles):
                        progress(tiles/total_tiles, f'Dựng ảnh: {tiles}/{total_tiles} vùng')
            if bad:
                if cleanup:
                    cleanup()
                del output
                continue
            if isinstance(output, np.memmap):
                output.flush()
            return RenderResult(output, bool(auto_crop), height*width > natural[0]*natural[1]*1.01,
                                (left, top, right, bottom), natural, tiles)
        except Exception:
            if cleanup:
                cleanup()
            raise
    raise HomographyError('Precise coverage could not produce a closed rectangle. Choose different images or disable closed cropping to inspect the boundary.')
