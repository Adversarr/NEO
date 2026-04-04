from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspec import solve_lowest_generalized_eigenpairs


def make_1d_laplacian(n: int) -> sp.csc_matrix:
    return sp.diags(
        diagonals=[
            -np.ones(n - 1),
            2.0 * np.ones(n),
            -np.ones(n - 1),
        ],
        offsets=[-1, 0, 1],
        format="csc",
    )


def make_identity_mass(n: int) -> sp.csc_matrix:
    return sp.eye(n, format="csc")


def align_eigenvectors(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    aligned = candidate.copy()
    for j in range(reference.shape[1]):
        if np.dot(reference[:, j], aligned[:, j]) < 0.0:
            aligned[:, j] *= -1.0
    return aligned


def main() -> None:
    n = 50
    k = 6

    L = make_1d_laplacian(n)
    M = make_identity_mass(n)

    result = solve_lowest_generalized_eigenpairs(L, M, k, sigma=0.0)
    scipy_evals, scipy_evecs = spla.eigsh(L, k=k, M=M, sigma=0.0, which="LM")

    pyspec_order = np.argsort(result.eigenvalues)
    scipy_order = np.argsort(scipy_evals)

    pyspec_evals = result.eigenvalues[pyspec_order]
    pyspec_evecs = result.eigenvectors[:, pyspec_order]
    scipy_evals = scipy_evals[scipy_order]
    scipy_evecs = scipy_evecs[:, scipy_order]
    pyspec_evecs = align_eigenvectors(scipy_evecs, pyspec_evecs)

    eval_err = np.max(np.abs(pyspec_evals - scipy_evals))
    evec_err = np.max(np.abs(pyspec_evecs - scipy_evecs))

    np.testing.assert_allclose(pyspec_evals, scipy_evals, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(pyspec_evecs, scipy_evecs, rtol=1e-6, atol=1e-8)

    print(f"status: {result.status}")
    print(f"converged: {result.nconv}/{k}")
    print("pyspec eigenvalues:")
    print(pyspec_evals)
    print("scipy eigenvalues:")
    print(scipy_evals)
    print(f"max eigenvalue abs error: {eval_err:.3e}")
    print(f"max eigenvector abs error: {evec_err:.3e}")
    print("verification: passed")


if __name__ == "__main__":
    main()
