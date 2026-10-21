"""Run and save the fixed-source mask optimization example."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from imaging import ImagingModel
from optimization import gradient_descent


def make_target(image_size=231):
    """The original four touching 30-pixel squares form one 60-pixel square."""
    if image_size < 61:
        raise ValueError("the example target needs an image_size of at least 61")
    target = np.zeros((image_size, image_size))
    center = image_size // 2
    target[center - 30:center + 30, center - 30:center + 30] = 1
    return target


def save_figure(path, target, initial_aerial, result, final_aerial, pixel_size_nm):
    from matplotlib import pyplot as plt

    edge = target.shape[0] * pixel_size_nm / 2
    extent = [-edge, edge, -edge, edge]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    intensity_max = max(1, final_aerial.max(), initial_aerial.max())
    mask_max = max(1, np.abs(result.mask).max())
    error_max = max(0.01, np.abs(final_aerial - target).max())
    images = (
        (target, "Target intensity", "gray", 0, 1),
        (initial_aerial, "Initial aerial intensity", "viridis", 0, intensity_max),
        (final_aerial, "Optimized aerial intensity", "viridis", 0, intensity_max),
        (result.mask, "Optimized mask amplitude", "RdBu_r", -mask_max, mask_max),
        (final_aerial - target, "Final intensity error", "RdBu_r", -error_max, error_max),
    )
    for axis, (values, title, cmap, vmin, vmax) in zip(axes.flat, images):
        plot = axis.imshow(values, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set(title=title, xlabel="x (nm)", ylabel="y (nm)")
        figure.colorbar(plot, ax=axis, shrink=0.8)
    axes[1, 2].semilogy(result.loss_history)
    axes[1, 2].set(title="Accepted objective values", xlabel="Accepted iteration", ylabel="Sum of squared intensity error")
    figure.suptitle(f"Fixed-source continuous-tone mask optimization | {result.status}")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--initial-step", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/mask_optimization"))
    args = parser.parse_args(argv)
    model = ImagingModel()
    target = make_target(model.image_size)
    initial_mask = target.copy()
    initial_aerial = model.forward(initial_mask)
    started = perf_counter()
    result = gradient_descent(
        lambda mask: model.loss_and_gradient(mask, target), initial_mask,
        max_iterations=args.iterations, initial_step=args.initial_step,
    )
    elapsed = perf_counter() - started
    final_aerial = model.forward(result.mask)
    initial_loss, final_loss = float(result.loss_history[0]), float(result.loss_history[-1])
    metrics = {
        "status": result.status, "converged": result.converged,
        "accepted_iterations": result.iterations,
        "objective_evaluations": result.evaluations, "rejected_steps": result.rejected_steps,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_reduction_percent": 100 * (1 - final_loss / initial_loss),
        "initial_mse": initial_loss / target.size, "final_mse": final_loss / target.size,
        "final_gradient_max_abs": float(result.gradient_norm_history[-1]),
        "monotone_accepted_loss": bool(np.all(np.diff(result.loss_history) <= 0)),
        "final_aerial_matches_recorded_loss": bool(np.isclose(np.sum((final_aerial-target)**2), final_loss, rtol=1e-12, atol=1e-12)),
        "mask_min": float(result.mask.min()), "mask_max": float(result.mask.max()),
        "optimization_seconds": elapsed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    parameters = {
        name: getattr(model, name)
        for name in ("wavelength_nm", "numerical_aperture", "pixel_size_nm", "image_size", "kernel_size", "source_size", "inner_sigma", "outer_sigma")
    }
    parameters.update(iterations=args.iterations, initial_step=args.initial_step, numpy_version=np.__version__)
    (args.output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output / "arrays.npz", target=target, initial_mask=initial_mask,
        initial_aerial=initial_aerial, mask=result.mask, aerial=final_aerial,
        loss_history=result.loss_history, gradient_norm_history=result.gradient_norm_history,
        step_sizes=result.step_sizes,
    )
    save_figure(args.output / "summary.png", target, initial_aerial, result, final_aerial, model.pixel_size_nm)
    print(json.dumps(metrics, indent=2))
    return 1 if result.status == "line_search_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
