"""Independent frequency-space quadrature of a circular pupil."""

import numpy as np


def pupil_field(mask, pixel_size_nm, cutoff, frequency, samples=257):
    """Integrate the mask spectrum over the shifted pupil, without a PSF or FFT."""
    axis = np.arange(mask.shape[0]) * pixel_size_nm
    df = 2 * cutoff / samples
    pupil_axis = -cutoff + (np.arange(samples) + 0.5) * df
    pupil = np.hypot(pupil_axis[:, None], pupil_axis[None, :]) <= cutoff
    fy, fx = pupil_axis + frequency[0], pupil_axis + frequency[1]
    py = np.exp(-2j * np.pi * fy[:, None] * axis)
    px = np.exp(-2j * np.pi * fx[:, None] * axis)
    spectrum = py @ mask @ px.T * pixel_size_nm**2
    return py.conj().T @ (pupil * spectrum) @ px.conj() * df**2
