"""Scalar Abbe imaging with fixed source intensities and real mask amplitudes.

Lengths are in nm, frequencies in cycles/nm. Convolutions are zero padded.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import signal, special


def _readonly(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class ImagingModel:
    """Cache kernels once. Construct a new model to change optical parameters.

    Optional source_weights are nonnegative intensities on the same centered
    frequency lattice as the annulus. Weights are copied and normalized.
    Even source sizes sample half-integer frequencies; odd sizes include zero.
    """

    wavelength_nm: float = 193.0
    numerical_aperture: float = 1.25
    pixel_size_nm: float = 2.0
    image_size: int = 231
    kernel_size: int = 41
    source_size: int = 18
    inner_sigma: float = 0.89
    outer_sigma: float = 0.99
    source_weights: NDArray | None = field(default=None, repr=False)
    amplitude_response: NDArray = field(init=False, repr=False)
    source_frequencies: NDArray = field(init=False, repr=False)
    active_weights: NDArray = field(init=False, repr=False)
    kernels: tuple = field(init=False, repr=False)
    adjoint_kernels: tuple = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("wavelength_nm", "numerical_aperture", "pixel_size_nm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("image_size", "kernel_size", "source_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for a centered same-size convolution")
        if self.kernel_size > self.image_size:
            raise ValueError("kernel_size must not exceed image_size")

        spacing = 1.0 / (self.image_size * self.pixel_size_nm)
        axis = (np.arange(self.source_size) - (self.source_size - 1) / 2) * spacing
        fy, fx = np.meshgrid(axis, axis, indexing="ij")
        if self.source_weights is None:
            if not (0 <= self.inner_sigma < self.outer_sigma <= 1):
                raise ValueError("require 0 <= inner_sigma < outer_sigma <= 1")
            cutoff = self.numerical_aperture / self.wavelength_nm
            if axis[-1] < self.outer_sigma * cutoff:
                raise ValueError("source_size is too small to cover the requested annulus")
            sigma = np.hypot(fy, fx) / cutoff
            weights = ((sigma >= self.inner_sigma) & (sigma <= self.outer_sigma)).astype(float)
        else:
            if np.iscomplexobj(self.source_weights):
                raise ValueError("source_weights must be real intensities")
            weights = np.array(self.source_weights, dtype=float, copy=True)
            if weights.shape != (self.source_size, self.source_size):
                raise ValueError("source_weights must have shape (source_size, source_size)")
        total = weights.sum()
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.isfinite(total) or total <= 0:
            raise ValueError("source_weights must be finite, nonnegative, and nonempty")
        weights /= total
        active = weights > 0

        xy = (np.arange(self.kernel_size) - self.kernel_size // 2) * self.pixel_size_nm
        y, x = np.meshgrid(xy, xy, indexing="ij")
        radius = np.hypot(y, x)
        argument = 2 * np.pi * self.numerical_aperture * radius / self.wavelength_nm
        response = np.full_like(argument, 0.5)  # lim J_1(x)/x as x -> 0
        np.divide(special.j1(argument), argument, out=response, where=argument != 0)
        response[radius > self.kernel_size // 2 * self.pixel_size_nm] = 0
        response /= response.sum()

        frequencies = np.column_stack((fy[active], fx[active]))
        kernels = tuple(
            _readonly(response * np.exp(2j * np.pi * (source_y * y + source_x * x)))
            for source_y, source_x in frequencies
        )
        object.__setattr__(self, "source_weights", _readonly(weights))
        object.__setattr__(self, "amplitude_response", _readonly(response))
        object.__setattr__(self, "source_frequencies", _readonly(frequencies))
        object.__setattr__(self, "active_weights", _readonly(weights[active]))
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "adjoint_kernels", tuple(_readonly(np.conj(k[::-1, ::-1])) for k in kernels))

    def _image_array(self, value, name):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        array = np.asarray(value, dtype=float)
        if array.shape != (self.image_size, self.image_size):
            raise ValueError(f"{name} must have shape ({self.image_size}, {self.image_size})")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    def _fields(self, mask):
        return tuple(signal.fftconvolve(mask, kernel, mode="same") for kernel in self.kernels)

    def _intensity(self, fields):
        return sum(weight * np.abs(value) ** 2 for weight, value in zip(self.active_weights, fields))

    def forward(self, mask):
        """Return aerial intensity for a real, unconstrained amplitude mask."""
        mask = self._image_array(mask, "mask")
        return self._intensity(self._fields(mask))

    def loss(self, mask, target):
        """Sum of squared errors between aerial and target intensities."""
        target = self._image_array(target, "target")
        return float(np.sum((self.forward(mask) - target) ** 2))

    def loss_and_gradient(self, mask, target):
        """Return loss and its derivative with respect to real mask values.

        I = sum_s w_s |K_s m|^2, L = sum (I-target)^2,
        dL/dm = 4 sum_s w_s Re[K_s^H ((I-target) K_s m)].
        The adjoint reverses and conjugates the full complex kernel.
        """
        mask = self._image_array(mask, "mask")
        target = self._image_array(target, "target")
        fields = self._fields(mask)
        residual = self._intensity(fields) - target
        gradient = np.zeros_like(mask)
        for weight, value, adjoint in zip(self.active_weights, fields, self.adjoint_kernels):
            gradient += 4 * weight * signal.fftconvolve(residual * value, adjoint, mode="same").real
        return float(np.sum(residual ** 2)), gradient
