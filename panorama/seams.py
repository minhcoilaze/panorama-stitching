"""Custom Dinic maximum flow / minimum cut and low-resolution panorama seams."""
from __future__ import annotations
from collections import deque
import numpy as np


def minimum_cut(count: int, edges: list[tuple[int, int, float]], source: int,
                sink: int) -> tuple[float, np.ndarray]:
    """Directed capacities; iterative blocking-flow paths avoid recursion limits."""
    graph: list[list[list]] = [[] for _ in range(count)]
    for a, b, capacity in edges:
        if capacity < 0 or not np.isfinite(capacity):
            raise ValueError('Capacities must be finite and non-negative.')
        graph[a].append([b, float(capacity), len(graph[b])])
        graph[b].append([a, 0., len(graph[a])-1])
    flow = 0.
    while True:
        level = np.full(count, -1, dtype=np.int32)
        level[source] = 0
        queue = deque([source])
        while queue:
            a = queue.popleft()
            for b, capacity, _ in graph[a]:
                if capacity > 1e-9 and level[b] < 0:
                    level[b] = level[a]+1
                    queue.append(b)
        if level[sink] < 0:
            return flow, level >= 0
        pointer = np.zeros(count, dtype=np.int32)
        while True:
            node, path = source, []
            while node != sink:
                while pointer[node] < len(graph[node]):
                    edge = graph[node][pointer[node]]
                    if edge[1] > 1e-9 and level[edge[0]] == level[node]+1:
                        break
                    pointer[node] += 1
                if pointer[node] == len(graph[node]):
                    level[node] = -1
                    if not path:
                        break
                    node, edge_index = path.pop()
                    pointer[node] = edge_index+1
                else:
                    path.append((node, int(pointer[node])))
                    node = graph[node][pointer[node]][0]
            if node != sink:
                break
            amount = min(graph[a][i][1] for a, i in path)
            for a, i in path:
                b, _, reverse = graph[a][i]
                graph[a][i][1] -= amount
                graph[b][reverse][1] += amount
            flow += amount


def _neighbours(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    return (padded[:-2, 1:w+1] | padded[2:, 1:w+1] |
            padded[1:h+1, :-2] | padded[1:h+1, 2:])


def graph_cut_ownership(images: list[np.ndarray], masks: list[np.ndarray],
                        order: list[int]) -> np.ndarray:
    """Sequential binary graph cuts on a deliberately small seam canvas."""
    labels = np.full(masks[0].shape, -1, dtype=np.int16)
    composite = np.zeros_like(images[0], dtype=np.float32)
    for index in order:
        old, new = labels >= 0, masks[index]
        overlap = old & new
        take = new & ~old
        if overlap.any():
            coordinates = np.argwhere(overlap)
            numbers = np.full(overlap.shape, -1, dtype=np.int32)
            numbers[overlap] = np.arange(len(coordinates))
            source, sink = len(coordinates), len(coordinates)+1
            difference = np.mean(np.abs(composite-images[index].astype(np.float32)), axis=2)+1.
            old_boundary = overlap & _neighbours(old & ~new)
            new_boundary = overlap & _neighbours(new & ~old)
            edges = []
            h, w = overlap.shape
            # A cut is meaningful when each image contributes an exclusive area.
            if old_boundary.any() and new_boundary.any():
                for y, x in coordinates:
                    a = int(numbers[y, x])
                    if old_boundary[y, x]:
                        edges.append((source, a, 1e8))
                    if new_boundary[y, x]:
                        edges.append((a, sink, 1e8))
                    for dy, dx in ((1, 0), (0, 1)):
                        yy, xx = y+dy, x+dx
                        if yy < h and xx < w and overlap[yy, xx]:
                            b = int(numbers[yy, xx])
                            capacity = float(difference[y, x]+difference[yy, xx])
                            edges.extend(((a, b, capacity), (b, a, capacity)))
                _, reachable = minimum_cut(sink+1, edges, source, sink)
                take[overlap] = ~reachable[:source]
        labels[take] = index
        composite[take] = images[index][take]
    return labels


def feather_ownership(labels: np.ndarray, count: int, radius: int = 3) -> list[np.ndarray]:
    """Custom separable box-filtered seam ownership for narrow feather bands."""
    weights = []
    size = 2*radius+1
    for index in range(count):
        values = (labels == index).astype(np.float32)
        for axis in (0, 1):
            padding = [(0, 0), (0, 0)]
            padding[axis] = (radius, radius)
            padded = np.pad(values, padding, mode='edge')
            zeros = [(0, 0), (0, 0)]
            zeros[axis] = (1, 0)
            sums = np.pad(np.cumsum(padded, axis=axis), zeros)
            if axis == 0:
                values = (sums[size:]-sums[:-size])/size
            else:
                values = (sums[:, size:]-sums[:, :-size])/size
        weights.append(values)
    return weights
