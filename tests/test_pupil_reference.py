import unittest

import numpy as np
from numpy.testing import assert_allclose

from imaging import ImagingModel
from pupil_reference import pupil_field


class PupilReferenceTests(unittest.TestCase):
    def test_full_response_matches_independent_pupil_quadrature(self):
        weights = np.zeros((3, 3))
        weights[0, 2] = 1
        model = ImagingModel(
            image_size=13, pixel_size_nm=20, source_size=3, source_weights=weights
        )
        mask = np.zeros((13, 13))
        mask[3:9, 4:10] = 1
        cutoff = model.numerical_aperture / model.wavelength_nm
        actual = model._fields(mask)[0]
        errors = []
        for samples in (65, 257):
            expected = pupil_field(
                mask, model.pixel_size_nm, cutoff, model.source_frequencies[0], samples
            )
            errors.append(np.linalg.norm(actual - expected) / np.linalg.norm(actual))
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 0.002)

    def test_old_short_support_changes_the_optical_result(self):
        weights = np.ones((1, 1))
        full = ImagingModel(
            image_size=41, pixel_size_nm=10, source_size=1, source_weights=weights
        )
        truncated = ImagingModel(
            image_size=41,
            kernel_size=9,
            pixel_size_nm=10,
            source_size=1,
            source_weights=weights,
        )
        mask = np.ones((41, 41))
        expected = pupil_field(
            mask, 10, full.numerical_aperture / full.wavelength_nm, [0, 0], 257
        )
        full_error = np.linalg.norm(full._fields(mask)[0] - expected)
        short_error = np.linalg.norm(truncated._fields(mask)[0] - expected)
        self.assertLess(full_error, 0.01 * short_error)

    def test_mask_pixel_quadrature_converges_for_a_fixed_physical_opening(self):
        centers = []
        for dx, cells in ((20, 20), (10, 40), (5, 80)):
            model = ImagingModel(
                image_size=cells,
                pixel_size_nm=dx,
                source_size=1,
                source_weights=np.ones((1, 1)),
            )
            axis = (np.arange(cells) - (cells - 1) / 2) * dx
            mask = (
                (np.abs(axis[:, None]) < 100) & (np.abs(axis[None, :]) < 100)
            ).astype(float)
            # Average the central four pixels to compare the same physical region.
            centers.append(
                model.forward(mask)[
                    cells // 2 - 1 : cells // 2 + 1, cells // 2 - 1 : cells // 2 + 1
                ].mean()
            )
        self.assertLess(abs(centers[2] - centers[1]), abs(centers[1] - centers[0]))


if __name__ == "__main__":
    unittest.main()
