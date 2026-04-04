from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from . import pcdlp_matrix


def make_sphere_points(count: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def main() -> int:
    points = make_sphere_points()
    laplace = pcdlp_matrix(points, 2, nn=12, hs=2.0, rho=3.0, htype="ddr")
    row_sum_error = float(np.abs(np.asarray(laplace.sum(axis=1)).ravel()).max())
    symmetric_laplace = 0.5 * (laplace + laplace.T)
    eigenvalues = eigsh(symmetric_laplace, k=6, which="SM", return_eigenvectors=False)

    print(f"points: {points.shape[0]}")
    print(f"matrix shape: {laplace.shape}")
    print(f"nnz: {laplace.nnz}")
    print(f"max |row sum|: {row_sum_error:.6e}")
    print("smallest eigenvalues:", " ".join(f"{value:.6e}" for value in np.sort(eigenvalues)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
