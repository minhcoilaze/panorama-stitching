"""Fused scalar renderer, authored here; Numba only compiles our loops.

This is the same inverse projection / four-neighbour bilinear / gain / feather
formula as the NumPy reference in rendering.py, with no packaged image kernel.
"""
from __future__ import annotations
import math
try:
    from numba import njit
    ACCELERATED = True
except ImportError:
    ACCELERATED = False
    def njit(*args, **kwargs):
        return lambda function: function


@njit(cache=True, nogil=True)
def accumulate_tile(image, matrix, kind, parameters, x_axis, y_axis,
                    sin_axis, cos_axis, weight, weight_x, weight_y, gain,
                    numerator, denominator, fallback, covered):
    """Add one source to a tile. Parameters: h,w,f,f_ref,angle,sx,sy."""
    work_h, work_w, focal, reference_focal, center_angle, scale_x, scale_y = parameters
    source_h, source_w = image.shape[:2]
    weight_h, weight_w = weight.shape
    for row in range(len(y_axis)):
        y = y_axis[row]
        wy = min(max(weight_y[row], 0.), weight_h-1.)
        wy0 = int(math.floor(wy))
        wy1 = min(wy0+1, weight_h-1)
        ay = wy-wy0
        for column in range(len(x_axis)):
            x = x_axis[column]
            if kind == 0:
                z = matrix[2, 0]*x+matrix[2, 1]*y+matrix[2, 2]
                if abs(z) <= 1e-10:
                    continue
                u = (matrix[0, 0]*x+matrix[0, 1]*y+matrix[0, 2])/z
                v = (matrix[1, 0]*x+matrix[1, 1]*y+matrix[1, 2])/z
            else:
                if abs(x/reference_focal-center_angle) >= math.pi:
                    continue
                s, c, h = sin_axis[column], cos_axis[column], y/reference_focal
                z = matrix[0, 2]*s+matrix[1, 2]*h+matrix[2, 2]*c
                if z <= 1e-10:
                    continue
                u = focal*(matrix[0, 0]*s+matrix[1, 0]*h+matrix[2, 0]*c)/z+(work_w-1)/2
                v = focal*(matrix[0, 1]*s+matrix[1, 1]*h+matrix[2, 1]*c)/z+(work_h-1)/2
            if not (0 <= u <= work_w-1 and 0 <= v <= work_h-1):
                continue
            u, v = (u+.5)/scale_x-.5, (v+.5)/scale_y-.5
            if not (0 <= u <= source_w-1 and 0 <= v <= source_h-1):
                continue
            first = not covered[row, column]
            covered[row, column] = True
            wx = min(max(weight_x[column], 0.), weight_w-1.)
            wx0 = int(math.floor(wx))
            wx1 = min(wx0+1, weight_w-1)
            ax = wx-wx0
            alpha = ((1-ay)*((1-ax)*weight[wy0, wx0]+ax*weight[wy0, wx1])+
                     ay*((1-ax)*weight[wy1, wx0]+ax*weight[wy1, wx1]))
            if alpha <= 1e-8 and not first:
                continue
            u0, v0 = int(math.floor(u)), int(math.floor(v))
            u1, v1 = min(u0+1, source_w-1), min(v0+1, source_h-1)
            du, dv = u-u0, v-v0
            for channel in range(3):
                value = ((1-dv)*((1-du)*image[v0, u0, channel]+du*image[v0, u1, channel])+
                         dv*((1-du)*image[v1, u0, channel]+du*image[v1, u1, channel]))
                value = min(max(value*gain[channel], 0.), 255.)
                if first:
                    fallback[row, column, channel] = value
                if alpha > 1e-8:
                    numerator[row, column, channel] += value*alpha
            if alpha > 1e-8:
                denominator[row, column] += alpha
