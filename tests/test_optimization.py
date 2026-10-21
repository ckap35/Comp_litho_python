import unittest

import numpy as np
from numpy.testing import assert_allclose

from optimization import gradient_descent
from test_imaging import small_model


def quadratic(mask):
    return 10 * np.sum(mask ** 2), 20 * mask


class OptimizationTests(unittest.TestCase):
    def test_backtracking_rejects_oversized_steps_then_converges(self):
        initial = np.array([[1.0, -2.0]])
        result = gradient_descent(quadratic, initial, relative_loss_tolerance=0)
        self.assertGreater(result.rejected_steps, 0)
        self.assertTrue(result.converged)
        self.assertEqual(result.status, "gradient_tolerance")
        self.assertTrue(np.all(np.diff(result.loss_history) < 0))
        self.assertLess(result.loss_history[-1], 1e-12)
        self.assertEqual(len(result.loss_history), result.iterations + 1)
        self.assertAlmostEqual(quadratic(result.mask)[0], result.loss_history[-1])
        assert_allclose(initial, [[1, -2]])

    def test_failure_keeps_initial_state_and_is_not_convergence(self):
        initial = np.array([1.0])
        result = gradient_descent(quadratic, initial, max_backtracks=0)
        self.assertEqual(result.status, "line_search_failed")
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)
        assert_allclose(result.mask, initial)
        assert_allclose(result.loss_history, [10])

    def test_failure_after_acceptance_keeps_last_accepted_state(self):
        def double_well(x):
            return np.sum((x ** 2 - 1) ** 2), 4 * x * (x ** 2 - 1)
        result = gradient_descent(double_well, np.array([0.1]), max_backtracks=0)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.status, "line_search_failed")
        self.assertFalse(result.converged)
        assert_allclose(result.mask, [0.496])
        self.assertAlmostEqual(double_well(result.mask)[0], result.loss_history[-1])

    def test_nonfinite_trials_are_backtracked(self):
        def restricted(x):
            if np.max(np.abs(x)) > 2:
                return np.inf, np.full_like(x, np.nan)
            return quadratic(x)
        result = gradient_descent(restricted, np.array([1.0]), max_iterations=3)
        self.assertGreater(result.rejected_steps, 0)
        self.assertTrue(np.all(np.isfinite(result.loss_history)))
        self.assertLess(result.loss_history[-1], 10)

    def test_repeatability(self):
        first = gradient_descent(quadratic, np.array([1.0, 2.0]), max_iterations=12)
        second = gradient_descent(quadratic, np.array([1.0, 2.0]), max_iterations=12)
        np.testing.assert_array_equal(first.mask, second.mask)
        np.testing.assert_array_equal(first.loss_history, second.loss_history)
        np.testing.assert_array_equal(first.step_sizes, second.step_sizes)

    def test_iteration_limit_is_not_convergence(self):
        result = gradient_descent(quadratic, np.ones((2, 2)), max_iterations=0)
        self.assertEqual(result.status, "max_iterations")
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)

    def test_already_stationary_initial_state(self):
        result = gradient_descent(quadratic, np.zeros((2, 2)))
        self.assertTrue(result.converged)
        self.assertEqual(result.status, "gradient_tolerance")
        self.assertEqual(result.evaluations, 1)

    def test_invalid_settings_and_initial_values(self):
        for settings in ({"initial_step": 0}, {"backtrack_factor": 1}, {"max_iterations": -1}, {"max_backtracks": 1.5}, {"gradient_tolerance": np.nan}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                gradient_descent(quadratic, np.ones((2, 2)), **settings)
        with self.assertRaises(ValueError):
            gradient_descent(quadratic, np.array([np.nan]))

    def test_imaging_optimization_reduces_loss_and_returns_matching_mask(self):
        model = small_model(asymmetric=True)
        target = np.zeros((17, 17))
        target[5:12, 5:12] = 1
        objective = lambda mask: model.loss_and_gradient(mask, target)
        result = gradient_descent(objective, target, max_iterations=40)
        self.assertLess(result.loss_history[-1], result.loss_history[0] * 0.8)
        self.assertTrue(np.all(np.diff(result.loss_history) <= 0))
        self.assertNotEqual(result.status, "line_search_failed")
        self.assertAlmostEqual(model.loss(result.mask, target), result.loss_history[-1], places=12)


if __name__ == "__main__":
    unittest.main()
