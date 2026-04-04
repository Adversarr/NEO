from __future__ import annotations

import unittest

import numpy as np

from pypcdlaplace import pcdlp_matrix
from pypcdlaplace.demo import make_sphere_points


class PcdLaplaceTests(unittest.TestCase):
    def test_returns_sparse_square_matrix(self) -> None:
        points = make_sphere_points(96, seed=1)
        matrix = pcdlp_matrix(points, 2, nn=12)
        self.assertEqual(matrix.shape, (96, 96))
        self.assertGreater(matrix.nnz, 0)
        self.assertLess(np.abs(np.asarray(matrix.sum(axis=1)).ravel()).max(), 1e-8)

    def test_invalid_tdim_raises(self) -> None:
        points = make_sphere_points(32, seed=2)
        with self.assertRaises(ValueError):
            pcdlp_matrix(points, 4)

    def test_small_regression_snapshot(self) -> None:
        points = make_sphere_points(48, seed=3)
        matrix = pcdlp_matrix(points, 2, nn=10)
        self.assertEqual(matrix.nnz, 2304)
        dense_slice = matrix[:3, :3].toarray()
        expected = np.array(
            [
                [-0.86960384, 0.09909161, 0.01067818],
                [0.00102283, -0.85622893, 0.04306136],
                [0.00197525, 0.02641219, -0.86448571],
            ]
        )
        np.testing.assert_allclose(dense_slice, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
