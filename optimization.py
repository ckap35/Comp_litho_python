"""Deterministic gradient descent with an Armijo backtracking line search."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class OptimizationResult:
    mask: NDArray
    loss_history: NDArray
    gradient_norm_history: NDArray
    step_sizes: NDArray
    status: str
    converged: bool
    evaluations: int
    rejected_steps: int

    @property
    def iterations(self):
        return len(self.step_sizes)


def gradient_descent(
    objective,
    initial_mask,
    *,
    max_iterations=500,
    initial_step=1.0,
    backtrack_factor=0.5,
    armijo_constant=1e-4,
    max_backtracks=40,
    gradient_tolerance=1e-6,
    relative_loss_tolerance=1e-9,
):
    """Minimize an objective returning (loss, gradient).

    An accepted step updates the mask, loss, and gradient together. A failed
    line search returns the last accepted state. The relative-loss criterion
    uses successive accepted loss values.
    """
    for name, value in (("max_iterations", max_iterations), ("max_backtracks", max_backtracks)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not np.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("initial_step must be finite and positive")
    if not 0 < backtrack_factor < 1 or not 0 < armijo_constant < 1:
        raise ValueError("backtrack_factor and armijo_constant must be between 0 and 1")
    for name, value in (("gradient_tolerance", gradient_tolerance), ("relative_loss_tolerance", relative_loss_tolerance)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if np.iscomplexobj(initial_mask):
        raise ValueError("initial_mask must be real")
    mask = np.array(initial_mask, dtype=float, copy=True)
    if mask.size == 0 or not np.all(np.isfinite(mask)):
        raise ValueError("initial_mask must be nonempty and finite")

    evaluations = 0
    rejected_steps = 0

    def evaluate(candidate):
        nonlocal evaluations
        value, gradient = objective(candidate)
        evaluations += 1
        if np.iscomplexobj(value) or np.ndim(value) != 0 or np.iscomplexobj(gradient):
            raise ValueError("objective must return a real scalar loss and real gradient")
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != mask.shape:
            raise ValueError("objective gradient must have the same shape as the mask")
        return float(value), gradient

    loss, gradient = evaluate(mask)
    if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise ValueError("initial objective value and gradient must be finite")
    losses = [loss]
    norms = [float(np.max(np.abs(gradient)))]
    steps = []
    status = "max_iterations"
    converged = False

    for _ in range(max_iterations):
        if norms[-1] <= gradient_tolerance:
            status, converged = "gradient_tolerance", True
            break
        step = initial_step
        slope = float(np.sum(gradient ** 2))
        for _ in range(max_backtracks + 1):
            trial_mask = mask - step * gradient
            trial_loss, trial_gradient = evaluate(trial_mask)
            valid = np.isfinite(trial_loss) and np.all(np.isfinite(trial_gradient))
            if valid and trial_loss <= loss - armijo_constant * step * slope:
                break
            rejected_steps += 1
            step *= backtrack_factor
        else:
            status = "line_search_failed"
            break

        previous_loss = loss
        mask, loss, gradient = trial_mask, trial_loss, trial_gradient
        losses.append(loss)
        norms.append(float(np.max(np.abs(gradient))))
        steps.append(step)
        if norms[-1] <= gradient_tolerance:
            status, converged = "gradient_tolerance", True
            break
        if previous_loss - loss <= relative_loss_tolerance * max(1.0, abs(previous_loss)):
            status, converged = "relative_loss_tolerance", True
            break

    return OptimizationResult(
        mask=mask, loss_history=np.asarray(losses),
        gradient_norm_history=np.asarray(norms), step_sizes=np.asarray(steps),
        status=status, converged=converged, evaluations=evaluations,
        rejected_steps=rejected_steps,
    )
