"""Small robust Levenberg--Marquardt solver, implemented with NumPy only.

No library optimizer is used. Finite differences, Huber IRLS weights and
diagonally scaled damping are explicit so the camera fit can be inspected.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(slots=True)
class OptimizationResult:
    parameters: np.ndarray
    cost: float
    initial_cost: float
    iterations: int


def robust_least_squares(residual: Callable[[np.ndarray], np.ndarray], initial: np.ndarray,
                         max_iterations: int = 30, huber_delta: float = 2.0) -> OptimizationResult:
    x = np.asarray(initial, dtype=np.float64).copy()
    if huber_delta <= 0 or not np.all(np.isfinite(x)):
        raise ValueError('Invalid optimizer parameters.')

    def cost(r):
        if not np.all(np.isfinite(r)):
            return float('inf')
        a = np.abs(r)
        return float(np.sum(np.where(a <= huber_delta, .5*r*r,
                                     huber_delta*(a-.5*huber_delta))))

    r = np.asarray(residual(x), dtype=np.float64).ravel()
    value = initial_cost = cost(r)
    if not np.isfinite(value):
        raise ValueError('The initial residual is not finite.')
    damping, iterations = 1e-3, 0
    for iterations in range(1, max_iterations+1):
        jac = np.empty((r.size, x.size), dtype=np.float64)
        for j in range(x.size):
            step = 1e-6 * max(1., abs(x[j]))
            shifted = x.copy()
            shifted[j] += step
            jac[:, j] = (np.asarray(residual(shifted)).ravel()-r)/step
        if not np.all(np.isfinite(jac)):
            break
        weights = np.sqrt(np.minimum(1., huber_delta/np.maximum(np.abs(r), 1e-12)))
        weighted = jac * weights[:, None]
        normal = weighted.T @ weighted
        gradient = weighted.T @ (r*weights)
        diagonal = np.maximum(np.diag(normal), 1e-8)
        accepted = False
        for _ in range(10):
            try:
                update = np.linalg.solve(normal+damping*np.diag(diagonal), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            candidate = x+update
            candidate_r = np.asarray(residual(candidate), dtype=np.float64).ravel()
            candidate_cost = cost(candidate_r)
            if candidate_cost < value:
                improvement = value-candidate_cost
                x, r, value = candidate, candidate_r, candidate_cost
                damping = max(damping/3, 1e-10)
                accepted = True
                break
            damping *= 10
        if not accepted or np.linalg.norm(update) < 1e-8*(1+np.linalg.norm(x)):
            break
        if improvement < 1e-9*(1+value):
            break
    return OptimizationResult(x, value, initial_cost, iterations)
