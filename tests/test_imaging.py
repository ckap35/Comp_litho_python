import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy import signal, special

from imaging import ImagingModel


def small_model(image_size=17, asymmetric=False):
    source = None
    if asymmetric:
        source = np.zeros((7, 7))
        source[1, 4] = 2
        source[4, 3] = 1
    return ImagingModel(
        image_size=image_size,
        source_size=7,
        pixel_size_nm=20,
        inner_sigma=0,
        outer_sigma=0.9,
        source_weights=source,
    )


class ImagingTests(unittest.TestCase):
    def test_response_center_limit_symmetry_and_physical_scaling(self):
        model = ImagingModel()
        h = model.amplitude_response
        center = model.kernel_size // 2
        argument = (
            2
            * np.pi
            * model.numerical_aperture
            * model.pixel_size_nm
            / model.wavelength_nm
        )
        expected_ratio = 0.5 / (special.j1(argument) / argument)
        self.assertAlmostEqual(
            h[center, center] / h[center, center + 1], expected_ratio, places=13
        )
        self.assertEqual(np.unravel_index(np.argmax(h), h.shape), (center, center))
        assert_allclose(h, h[::-1, ::-1], atol=1e-15)
        expected_center = (
            np.pi
            * (model.numerical_aperture / model.wavelength_nm) ** 2
            * model.pixel_size_nm**2
        )
        self.assertAlmostEqual(h[center, center], expected_center, places=14)
        self.assertEqual(model.kernel_size, 2 * model.image_size - 1)

    def test_annular_source_center_and_sigma(self):
        model = ImagingModel()
        locations = np.argwhere(model.source_weights > 0)
        assert_allclose(locations.mean(axis=0), [8.5, 8.5], atol=1e-15)
        expected_frequencies = (locations - 8.5) / (231 * 2)
        assert_allclose(model.source_frequencies, expected_frequencies, atol=1e-15)
        sigma = np.linalg.norm(model.source_frequencies, axis=1) / (1.25 / 193)
        self.assertTrue(np.all((sigma >= 0.89) & (sigma <= 0.99)))
        self.assertAlmostEqual(model.source_weights.sum(), 1.0)

    def test_source_phase_matches_physical_frequency(self):
        model = small_model(asymmetric=True)
        center = model.kernel_size // 2
        h = model.amplitude_response
        for frequency, kernel in zip(model.source_frequencies, model.kernels):
            phase_step_y = kernel[center + 1, center] / h[center + 1, center]
            phase_step_x = kernel[center, center + 1] / h[center, center + 1]
            assert_allclose(
                phase_step_y,
                np.exp(2j * np.pi * frequency[0] * model.pixel_size_nm),
                atol=1e-14,
            )
            assert_allclose(
                phase_step_x,
                np.exp(2j * np.pi * frequency[1] * model.pixel_size_nm),
                atol=1e-14,
            )

    def test_disk_source_includes_on_axis_sample(self):
        model = small_model()
        self.assertGreater(model.source_weights[3, 3], 0)

    def test_forward_matches_direct_convolution(self):
        rng = np.random.default_rng(3)
        for size in (16, 17):
            for asymmetric in (False, True):
                with self.subTest(size=size, asymmetric=asymmetric):
                    model = small_model(size, asymmetric)
                    mask = rng.normal(size=(size, size))
                    expected = sum(
                        weight
                        * np.abs(
                            signal.convolve2d(
                                mask, kernel, mode="same", boundary="fill"
                            )
                        )
                        ** 2
                        for weight, kernel in zip(model.active_weights, model.kernels)
                    )
                    assert_allclose(
                        model.forward(mask), expected, rtol=1e-12, atol=1e-13
                    )

    def test_complex_convolution_adjoint_including_boundaries(self):
        rng = np.random.default_rng(4)
        for size in (16, 17):
            model = small_model(size, asymmetric=True)
            x = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
            y = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
            for kernel, adjoint in zip(model.kernels, model.adjoint_kernels):
                forward = signal.convolve2d(x, kernel, mode="same")
                backward = signal.convolve2d(y, adjoint, mode="same")
                assert_allclose(
                    np.vdot(forward, y), np.vdot(x, backward), rtol=1e-12, atol=1e-12
                )

    def test_directional_derivatives_with_symmetric_and_asymmetric_sources(self):
        rng = np.random.default_rng(5)
        for size in (16, 17):
            for asymmetric in (False, True):
                model = small_model(size, asymmetric)
                mask = rng.uniform(-0.3, 1.2, size=(size, size))
                target = rng.uniform(size=(size, size))
                value, gradient = model.loss_and_gradient(mask, target)
                self.assertAlmostEqual(value, model.loss(mask, target), places=12)
                for _ in range(3):
                    direction = rng.normal(size=mask.shape)
                    direction /= np.linalg.norm(direction)
                    expected = np.sum(gradient * direction)
                    for epsilon in (1e-3, 1e-4, 1e-5):
                        numerical = (
                            model.loss(mask + epsilon * direction, target)
                            - model.loss(mask - epsilon * direction, target)
                        ) / (2 * epsilon)
                        with self.subTest(
                            size=size, asymmetric=asymmetric, epsilon=epsilon
                        ):
                            assert_allclose(numerical, expected, rtol=1e-6, atol=1e-8)

    def test_pixel_derivatives_at_corners_edges_and_center(self):
        rng = np.random.default_rng(6)
        model = small_model(asymmetric=True)
        mask = rng.uniform(size=(17, 17))
        target = rng.uniform(size=mask.shape)
        _, gradient = model.loss_and_gradient(mask, target)
        for row, column in (
            (0, 0),
            (0, 16),
            (16, 0),
            (16, 16),
            (0, 8),
            (8, 0),
            (8, 8),
            (3, 12),
        ):
            delta = np.zeros_like(mask)
            delta[row, column] = 1e-5
            numerical = (
                model.loss(mask + delta, target) - model.loss(mask - delta, target)
            ) / 2e-5
            assert_allclose(numerical, gradient[row, column], rtol=1e-6, atol=1e-8)

    def test_single_pixel_kernel_has_exact_intensity_and_gradient(self):
        model = ImagingModel(
            image_size=7, kernel_size=1, source_size=1, source_weights=np.ones((1, 1))
        )
        rng = np.random.default_rng(7)
        mask, target = rng.normal(size=(7, 7)), rng.uniform(size=(7, 7))
        value, gradient = model.loss_and_gradient(mask, target)
        gain = (
            np.pi
            * (model.numerical_aperture / model.wavelength_nm) ** 2
            * model.pixel_size_nm**2
        ) ** 2
        assert_allclose(model.forward(mask), gain * mask**2, atol=1e-13)
        self.assertAlmostEqual(value, np.sum((gain * mask**2 - target) ** 2), places=10)
        assert_allclose(
            gradient,
            4 * gain * mask * (gain * mask**2 - target),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_zero_padding_does_not_wrap_corner_impulse(self):
        model = small_model()
        mask = np.zeros((17, 17))
        mask[0, 0] = 1
        aerial = model.forward(mask)
        self.assertGreater(aerial[0, 0], 0)
        center = model.kernel_size // 2
        expected = sum(
            weight * np.abs(kernel[center : center + 17, center : center + 17]) ** 2
            for weight, kernel in zip(model.active_weights, model.kernels)
        )
        assert_allclose(aerial, expected, rtol=1e-11, atol=1e-17)
        self.assertLess(aerial[-1, -1], aerial[0, 0] / 100)

    def test_optics_and_caches_are_fixed_during_evaluation(self):
        model = small_model()
        kernel_ids = tuple(map(id, model.kernels))
        with patch("imaging.special.j1", side_effect=AssertionError("kernel rebuilt")):
            model.forward(np.ones((17, 17)))
            model.loss_and_gradient(np.ones((17, 17)), np.zeros((17, 17)))
        self.assertEqual(tuple(map(id, model.kernels)), kernel_ids)
        with self.assertRaises(FrozenInstanceError):
            model.wavelength_nm = 200
        with self.assertRaises(ValueError):
            model.source_weights[0, 0] = 1

    def test_custom_source_is_copied_and_normalized(self):
        weights = np.ones((7, 7))
        model = ImagingModel(
            image_size=17, kernel_size=5, source_size=7, source_weights=weights
        )
        weights[:] = 0
        assert_allclose(model.source_weights, np.full((7, 7), 1 / 49))

    def test_invalid_optics_and_sources_are_rejected(self):
        invalid = (
            {"wavelength_nm": 0},
            {"pixel_size_nm": np.nan},
            {"kernel_size": 4},
            {"source_size": 0},
            {"inner_sigma": 1},
            {"source_size": 1},
            {"source_weights": np.zeros((18, 18))},
            {"source_weights": -np.ones((18, 18))},
            {"source_weights": np.ones((2, 2))},
            {"source_weights": np.full((18, 18), np.nan)},
            {"source_weights": np.ones((18, 18), dtype=complex)},
        )
        for settings in invalid:
            with self.subTest(settings=list(settings)), self.assertRaises(ValueError):
                ImagingModel(**settings)

    def test_invalid_masks_are_rejected(self):
        model = small_model()
        for mask in (
            np.ones((3, 3)),
            np.full((17, 17), np.nan),
            np.ones((17, 17), dtype=complex),
        ):
            with self.assertRaises(ValueError):
                model.forward(mask)


if __name__ == "__main__":
    unittest.main()
