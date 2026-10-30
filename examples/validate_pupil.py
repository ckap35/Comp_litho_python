"""Compare the imaging operator with direct quadrature over the pupil."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imaging import ImagingModel
from pupil_reference import pupil_field


def main():
    output = ROOT / "docs/figures"
    output.mkdir(parents=True, exist_ok=True)
    model = ImagingModel(
        image_size=41, pixel_size_nm=10, source_size=1, source_weights=np.ones((1, 1))
    )
    mask = np.zeros((41, 41))
    mask[10:31, 10:31] = 1
    reference = pupil_field(
        mask, 10, model.numerical_aperture / model.wavelength_nm, [0, 0], samples=513
    )
    intensity = np.abs(reference) ** 2
    actual = model.forward(mask)
    short = ImagingModel(
        image_size=41,
        pixel_size_nm=10,
        source_size=1,
        source_weights=np.ones((1, 1)),
        kernel_size=9,
    )
    short_image = short.forward(mask)
    errors = {}
    for label, values in (("full", actual), ("9_by_9_kernel", short_image)):
        errors[label] = float(
            np.linalg.norm(values - intensity) / np.linalg.norm(intensity)
        )
    x = (np.arange(41) - 20) * 10
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for ax, values, title in zip(
        axes[:2], (mask, actual), ("Mask amplitude", "Aerial intensity")
    ):
        im = ax.imshow(values, origin="lower", extent=(-205, 205, -205, 205))
        ax.set(title=title, xlabel="x (nm)", ylabel="y (nm)")
        fig.colorbar(im, ax=ax)
    axes[2].plot(x, intensity[20], "k--", label="Pupil quadrature")
    axes[2].plot(x, actual[20], label="Full response")
    axes[2].plot(x, short_image[20], label="9×9 kernel (±40 nm)")
    axes[2].set(title="Central line cut", xlabel="x (nm)", ylabel="Intensity")
    axes[2].legend(fontsize=8)
    fig.savefig(output / "pupil_validation.png", dpi=160)
    plt.close(fig)
    metrics = {
        "relative_intensity_errors": errors,
        "reference_pupil_samples_per_axis": 513,
        "image_size": 41,
        "pixel_size_nm": 10,
        "full_kernel_size": model.kernel_size,
    }
    (ROOT / "docs/pupil_validation.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
