"""Largest all-valid rectangle: histogram + monotonic stack, O(H*W)."""
from __future__ import annotations
import numpy as np


def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return top,left,bottom,right (exclusive) without treating black as empty."""
    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2 or not valid.size:
        raise ValueError('Expected a non-empty 2-D coverage mask.')
    heights = np.zeros(valid.shape[1]+1, dtype=np.int64)
    best, rectangle = 0, (0, 0, 0, 0)
    for row in range(valid.shape[0]):
        heights[:-1] = np.where(valid[row], heights[:-1]+1, 0)
        stack: list[int] = []
        for column in range(len(heights)):
            while stack and heights[stack[-1]] > heights[column]:
                height = int(heights[stack.pop()])
                left = stack[-1]+1 if stack else 0
                area = height*(column-left)
                if area > best:
                    best = area
                    rectangle = (row-height+1, left, row+1, column)
            stack.append(column)
    if not best:
        raise ValueError('No valid image area remains to crop.')
    return rectangle


def erode_valid_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """One-pixel 8-neighbour erosion, including the outside border as invalid."""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, constant_values=False)
        out = np.logical_and.reduce([padded[y:y+out.shape[0], x:x+out.shape[1]]
                                     for y in range(3) for x in range(3)])
    return out
