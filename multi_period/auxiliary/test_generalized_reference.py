"""A cross-implementation invariant for the generalized scenario integrator."""
import unittest

from run_robustness import BASE, FORWARD, INNER, ORDER, generalized_value
from factor_model import value


class GeneralizedReferenceTest(unittest.TestCase):
    def test_common_penalties_reproduce_original_two_period_value(self):
        generalized = generalized_value(FORWARD[:2], BASE, (100.0,) * 4)
        original = value(FORWARD[:2], BASE, order=ORDER, inner=INNER)
        self.assertAlmostEqual(generalized, original, delta=0.001)


if __name__ == "__main__":
    unittest.main()
